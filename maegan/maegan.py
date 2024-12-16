import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from maegan.gan import Generator, Discriminator, Encoder
from maegan.tools import SimpleDataset, load_UGR16, NormalizeTransform
from maegan.mae import MAE
import numpy as np
import pandas as pd
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.model_zoo import tqdm
import pickle

class MAEGAN:

    def __init__(self, config):
        # Parameters:
        self.device = config["device"]

        self.model_dir = os.path.join(config["model_dir"], "maegan")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "mae"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "gan"), exist_ok=True)

        config = config["maegan"]
        self.mae_model = MAE(config["in_dim"],config["maxAE"],0,0, feature_map=None)
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.in_dim = config["in_dim"]
        self.maxAE = config["maxAE"]
        self.lr = config["lr"]
        self.b1 = config["b1"]
        self.b2 = config["b2"]
        self.n_critic = config["n_critic"]
        self.feature_map = None
        self.generator = None
        self.discriminator = None
        self.encoder = None
        self.gan_input_dim = None


    def train(self, data):
        print("Running KitNET:")
        mae_output = self.trainMAE(data)
        print("Running fanogan:")
        train_dataloader = self.load_gan_input(mae_output)
        self.gan_input_dim = mae_output.shape[1]
        latent_dim = int(self.gan_input_dim * 0.5)

        self.generator = Generator(self.gan_input_dim, latent_dim)
        self.discriminator = Discriminator(self.gan_input_dim)
        self.train_wgangp(train_dataloader, "cpu", latent_dim)

        self.encoder = Encoder(self.gan_input_dim, latent_dim)
        self.train_encoder_izif(train_dataloader, "cpu")
        self.save()

    def detect(self, data, output_dir):
        mae_output = self.testMAE(data)
        test_dataloader = self.load_gan_input(mae_output)
        return self.test_anomaly_detection(test_dataloader, "cpu", output_dir)

    def load_gan_input(self, mae_output):
        mae_output = torch.from_numpy(mae_output).float()
        batch_size = self.batch_size
        dataset = SimpleDataset(mae_output, np.zeros(len(mae_output)), transform=None)
        train_dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        return train_dataloader

    def trainMAE(self, data):
        if self.feature_map == None:
            for i in tqdm(range(data.shape[0])):
                self.mae_model.trainfm(data[i,]) #will train during the grace periods, then execute on all the rest.
            self.feature_map = self.mae_model.cluster()
            gan_input_dim = len(self.feature_map)
        print(self.feature_map)
        print(gan_input_dim)
        output = np.zeros([data.shape[0], gan_input_dim]) # a place to save the scores
        for i in tqdm(range(data.shape[0])):
            output[i] = self.mae_model.train(data[i,]) #will train during the grace periods, then execute on all the rest.
        self.mae_model.save(os.path.join(self.model_dir, "mae"))
        return output

    def testMAE(self, data):
        self.mae_model = MAE.load(os.path.join(self.model_dir, "mae"))
        output = np.zeros([data.shape[0], self.gan_input_dim])
        for i in tqdm(range(data.shape[0])):
            output[i] = self.mae_model.execute(data[i,])
        return output
    
    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        
        alpha = torch.rand(real_samples.shape[1], device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(*d_interpolates.shape, device=device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=fake, create_graph=True,
                                retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def train_wgangp(self, dataloader, device, latent_dim, lambda_gp=10):
        self.generator.to(device)
        self.discriminator.to(device)

        optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                    lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.lr, betas=(self.b1, self.b2))

        padding_epoch = len(str(self.n_epochs))
        padding_i = len(str(len(dataloader)))

        batches_done = 0
        for epoch in range(self.n_epochs):
            for i, (gsa_input, _)in enumerate(dataloader):
                real_imgs = gsa_input.to(device)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.randn(gsa_input.shape[0], latent_dim, device=device)

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(
                                                            real_imgs.data,
                                                            fake_imgs.data,
                                                            device)
                # Adversarial loss
                d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                        + lambda_gp * gradient_penalty)

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator and output log every n_critic steps
                if i % self.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    print(f"[Epoch {epoch:{padding_epoch}}/{self.n_epochs}] "
                        f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():3f}] "
                        f"[G loss: {g_loss.item():3f}]")

                    # if batches_done % opt.sample_interval == 0:
                    #     save_image(fake_imgs.data[:25],
                    #                f"results/images/{batches_done:06}.png",
                    #                nrow=5, normalize=True)

                    batches_done += self.n_critic

            torch.save(self.generator.state_dict(), os.path.join(self.model_dir, "gan", "generator"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, "gan", "discriminator"))

    def train_encoder_izif(self, dataloader, device, kappa=1.0):
        self.generator.load_state_dict(torch.load(os.path.join(self.model_dir, "gan", "generator")))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.model_dir, "gan", "discriminator")))

        self.generator.to(device).eval()
        self.discriminator.to(device).eval()
        self.encoder.to(device)

        criterion = nn.MSELoss()

        optimizer_E = torch.optim.Adam(self.encoder.parameters(),
                                    lr=self.lr, betas=(self.b1, self.b2))

        padding_epoch = len(str(self.n_epochs))
        padding_i = len(str(len(dataloader)))

        batches_done = 0
        for epoch in range(self.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = imgs.to(device)

                # ----------------
                #  Train Encoder
                # ----------------

                optimizer_E.zero_grad()

                # Generate a batch of latent variables
                z = self.encoder(real_imgs)

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real features
                real_features = self.discriminator.forward_features(real_imgs)
                # Fake features
                fake_features = self.discriminator.forward_features(fake_imgs)

                # izif architecture
                loss_imgs = criterion(fake_imgs, real_imgs)
                loss_features = criterion(fake_features, real_features)
                e_loss = loss_imgs + kappa * loss_features

                e_loss.backward()
                optimizer_E.step()

                # Output training log every n_critic steps
                if i % self.n_critic == 0:
                    print(f"[Epoch {epoch:{padding_epoch}}/{self.n_epochs}] "
                        f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                        f"[E loss: {e_loss.item():3f}]")

                    # if batches_done % opt.sample_interval == 0:
                    #     fake_z = encoder(fake_imgs)
                    #     reconfiguration_imgs = generator(fake_z)
                    #     save_image(reconfiguration_imgs.data[:25],
                    #                f"results/images_e/{batches_done:06}.png",
                    #                nrow=5, normalize=True)

                    batches_done += self.n_critic
            torch.save(self.encoder.state_dict(), os.path.join(self.model_dir, "gan", "encoder"))

    def test_anomaly_detection(self, dataloader, device, output_dir, kappa=1.0):
        gan_dir = os.path.join(self.model_dir, "gan")
        self.generator.load_state_dict(torch.load(os.path.join(gan_dir, "generator")))
        self.discriminator.load_state_dict(torch.load(os.path.join(gan_dir, "discriminator")))
        self.encoder.load_state_dict(torch.load(os.path.join(gan_dir, "encoder")))


        self.generator.to(device).eval()
        self.discriminator.to(device).eval()
        self.encoder.to(device).eval()

        criterion = nn.MSELoss()
        score_path = os.path.join(output_dir, "score.csv")
        with open(score_path, "w") as f:
            f.write("img_distance,anomaly_score,z_distance\n")
        results = []
        for (img, label) in tqdm(dataloader):

            real_img = img.to(device)

            real_z = self.encoder(real_img)
            fake_img = self.generator(real_z)
            fake_z = self.encoder(fake_img)

            real_feature = self.discriminator.forward_features(real_img)
            fake_feature = self.discriminator.forward_features(fake_img)

            # Scores for anomaly detection
            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + kappa * loss_feature

            z_distance = criterion(fake_z, real_z)

            with open(score_path, "a") as f:
                f.write(f"{img_distance},{anomaly_score},{z_distance}\n")
            results.append(anomaly_score.item())
        return results

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        # 保存 KitNET 的结构和超参数
        with open(os.path.join(self.model_dir, "maegan.pkl"), "wb") as f:
            pickle.dump(self, f)

        mae_dir = os.path.join(self.model_dir, "mae")
        self.mae_model.save(mae_dir)

        gan_dir = os.path.join(self.model_dir, "gan")
        torch.save(self.generator.state_dict(), os.path.join(gan_dir, "generator"))
        torch.save(self.discriminator.state_dict(), os.path.join(gan_dir, "discriminator"))
        torch.save(self.encoder.state_dict(), os.path.join(gan_dir, "encoder"))

    # 加载 KitNET 模型，包括所有 dA 实例的权重
    @staticmethod
    def load(config) -> "MAEGAN":
        model_dir = os.path.join(config["model_dir"], "maegan")

        with open(os.path.join(model_dir, "maegan.pkl"), "rb") as f:
            model = pickle.load(f)

        mae_dir = os.path.join(model_dir, "mae")
        model.mae_model.load(mae_dir)

        gan_dir = os.path.join(model_dir, "gan")
        model.generator.load_state_dict(torch.load(os.path.join(gan_dir, "generator")))
        model.discriminator.load_state_dict(torch.load(os.path.join(gan_dir, "discriminator")))
        model.encoder.load_state_dict(torch.load(os.path.join(gan_dir, "encoder")))
        
        return model