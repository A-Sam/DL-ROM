import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import torchvision
from model import AE_3D_Dataset, UNet_3D
import matplotlib.pyplot as plt
import imageio
import cv2

# from model import MyDataset, MLP_Dataset, LSTM_Dataset, autoencoder, autoencoder_B, MLP, Unet, LSTM, LSTM_B, AE_3D_Dataset, autoencoder_3D,UNet_3D
from train import training, validation, test, simulate
from utils import (
    load_transfer_learning,
    insert_time_channel,
    find_weight,
    load_transfer_learning_UNet_3D,
    save_loss,
    normalize_data,
    MSE,
    plot_training,
    principal_components_p1p2
)
import warnings
import pdb
import cv

# plot all PCA for all scenarios
def plotPCAAllScenarios():
    datasets = ["2d_airfoil","2d_cylinder_CFD","2d_sq_cyl","SST"]
    test_epoch = 100
    device = "cuda"
    for dataset_name in datasets:
        model = UNet_3D(name=dataset_name)
        model = model.to(device)
        
        ####
        if dataset_name == "2d_cylinder":
            u = np.load("../data/cylinder_u.npy", allow_pickle=True)[:-1, ...][
                :, :, 40:-280
            ]
            u = normalize_data(u)
            # v = np.load('../data/cylinder_v.npy', allow_pickle=True)[:-1, ...]

        elif dataset_name == "boussinesq":
            ux = np.load("../data/boussinesq_u.npy", allow_pickle=True)[:-1, ...][
                :, 50:-80, :
            ]
            u = np.array(
                [
                    cv2.resize(ux[i], (160, 320), interpolation=cv2.INTER_CUBIC)
                    for i in range(ux.shape[0])
                ]
            )
            u = normalize_data(u)
            # v = np.load('../data/boussinesq_v.npy', allow_pickle=True)[:-1, ...]

        elif dataset_name == "SST":
            u = np.load("../data/sea_surface_noaa.npy", allow_pickle=True)[:2000, ...][
                :, 10:-10, 20:-20
            ]
            u = normalize_data(u)

        elif dataset_name == "2d_cylinder_CFD":
            u_comp = np.load("../data/Vort100.npz", allow_pickle=True)
            # u_comp = np.load('../data/Velocity160.npz', allow_pickle=True)

            u_flat = u_comp["arr_0"]
            print("shape u_flat= " + str(u_flat.shape))
            u = u_flat.reshape(u_flat.shape[0], 320, 80)
            u = np.transpose(u, (0, 2, 1)).astype(np.float32)
            u = normalize_data(u)

        elif dataset_name == "2d_sq_cyl":
            u_flat = np.load("../data/sq_cyl_vort.npy", allow_pickle=True)  # sq_cyl_vel
            u = u_flat.reshape(u_flat.shape[0], 320, 80)
            u = np.transpose(u, (0, 2, 1)).astype(np.float32)[
                :2000, ...
            ]  # temporarily reducing dataset size
            u = normalize_data(u)

        elif dataset_name == "channel_flow":
            u = np.load("../data/channel_data_2500.npy", allow_pickle=True).astype(
                np.float32
            )
            u = normalize_data(u)

        elif dataset_name == "2d_airfoil":
            u_flat = np.load("../data/airfoil80x320_data.npy", allow_pickle=True)
            print(u_flat.shape)
            u = u_flat.reshape(u_flat.shape[0], 320, 80)
            u = np.transpose(u, (0, 2, 1))[:, :, 140:-20].astype(np.float32)
            u = normalize_data(u)

        elif dataset_name == "2d_plate":
            u_flat = np.load("../data/platekepsilon.npy", allow_pickle=True)
            print(u_flat.shape)
            u = u_flat.reshape(u_flat.shape[0], 360, 180)
            u = np.transpose(u, (0, 2, 1))[:, :-20, :-40].astype(np.float32)
            u = normalize_data(u)

        else:
            print("Dataset Not Found")

        print(f"Data Loaded in Dataset: {dataset_name} with shape {u.shape[0]}")

        # train/val split
        train_to_val = 0.75
        # rand_array = np.random.permutation(1500)
        # print(rand_array)

        u_train = u[: int(train_to_val * u.shape[0]), ...]
        u_validation = u[int(train_to_val * u.shape[0]) :, ...]

        print("Training Set Shape= " + str(u_train.shape))
        print("Validation Set Shape= " + str(u_validation.shape))

        # u = insert_time_channel(u, 10)
        # print(u.shape);

        img_transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5])
            ]
        )
        ####
        
        
        PATH = find_weight(dataset_name, test_epoch)

        print("Loading Scenario:" + PATH)
        model.load_state_dict(torch.load(PATH))

        test_dataset = AE_3D_Dataset(
            u_validation, dataset_name, transform=img_transform
        )
        test_loader_args = dict(batch_size=1, shuffle=False, num_workers=16)
        test_loader = data.DataLoader(test_dataset, **test_loader_args)

        labels, preds = test(model, test_loader)
        name = f"../results/{dataset_name}/labels.npy"
        np.save(name, labels)

        name = f"../results/{dataset_name}/predictions.npy"
        np.save(name, preds)

        # Flatten the images
        labels_flat = labels.reshape(labels.shape[0], -1)
        preds_flat = preds.reshape(preds.shape[0], -1)

        # Center the data by subtracting the mean
        labels_centered = labels_flat - np.mean(labels_flat, axis=0)
        preds_centered = preds_flat - np.mean(preds_flat, axis=0)

        # Perform SVD
        U_labels, S_labels, Vt_labels = np.linalg.svd(labels_centered, full_matrices=False)
        U_preds, S_preds, Vt_preds = np.linalg.svd(preds_centered, full_matrices=False)

        # Calculate the principal components projections
        p1_labels = U_labels[:, 0] * S_labels[0]
        p2_labels = U_labels[:, 1] * S_labels[1]

        p1_preds = U_preds[:, 0] * S_preds[0]
        p2_preds = U_preds[:, 1] * S_preds[1]

        # Calculate the differences in the projections
        p1_diff = p1_preds - p1_labels
        p2_diff = p2_preds - p2_labels


        # Plot the differences
        plt.scatter(p1_diff, p2_diff, alpha=0.5, s=5, label= dataset_name)
        # plt.plot(p1_diff, p2_diff, '-.', linewidth=0.1, color='red', alpha = 0.7)
        del model
    plt.xlabel('Difference in First Principal Component P1')
    plt.ylabel('Difference in Second Principal Component P2')
    plt.title('Ground Truth vs Predictions for Different Data Sets')
    plt.legend(loc='lower right')
    plt.grid(True, alpha= 0.25)
    plt.savefig(
            f"../results/all_scenarios_p1_p2_diff.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0)
    plt.close()




"""
python main.py -N 100 -B 32 -d_set 2d_cylinder_CFD --train/ --transfer/ --simulate --test/ -test_epoch 
"""

if __name__ == "__main__":
    # arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", dest="num_epochs", type=int, help="Number of Epochs")
    parser.add_argument(
        "-B", dest="batch_size", type=int, default=16, help="Batch Size"
    )
    parser.add_argument(
        "-d_set",
        dest="dataset",
        type=str,
        default="2d_cylinder_CFD",
        help="Name of Dataset",
    )
    parser.add_argument(
        "--test_epoch",
        dest="test_epoch",
        type=int,
        default=None,
        help="Epoch for testing",
    )
    parser.add_argument("--test", dest="testing", action="store_true")
    parser.add_argument("--train", dest="training", action="store_true")
    parser.add_argument("--transfer", dest="transfer", action="store_true")
    parser.add_argument("--simulate", dest="simulate", action="store_true")
    parser.add_argument("--last_pth", dest="last_pth", type=int, default=None)

    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dataset_name = args.dataset
    test_epoch = args.test_epoch
    transfer_learning = args.transfer
    last_pth = args.last_pth

    print(num_epochs, batch_size)

    if not os.path.exists(f"../results"):
        os.mkdir(f"../results")

    if not os.path.exists(f"../results/{dataset_name}"):
        os.mkdir(f"../results/{dataset_name}")

    if not os.path.exists(f"../simulate"):
        os.mkdir(f"../simulate")

    # Making folders to save reconstructed images, input images and weights
    if not os.path.exists(f"../results/{dataset_name}/"):
        os.mkdir(f"../results/{dataset_name}/")

    if not os.path.exists(f"../results/{dataset_name}/weights/"):
        os.mkdir(f"../results/{dataset_name}/weights/")

    if not os.path.exists(f"../results/{dataset_name}/p1p2_plots/"):
        os.mkdir(f"../results/{dataset_name}/p1p2_plots/")

    if not os.path.exists(f"../simulate/{dataset_name}"):
        os.mkdir(f"../simulate/{dataset_name}")

    warnings.filterwarnings("ignore")

    # Running the model on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training will be done on: " + device)
    if dataset_name == "2d_cylinder":
        u = np.load("../data/cylinder_u.npy", allow_pickle=True)[:-1, ...][
            :, :, 40:-280
        ]
        u = normalize_data(u)
        # v = np.load('../data/cylinder_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == "boussinesq":
        ux = np.load("../data/boussinesq_u.npy", allow_pickle=True)[:-1, ...][
            :, 50:-80, :
        ]
        u = np.array(
            [
                cv2.resize(ux[i], (160, 320), interpolation=cv2.INTER_CUBIC)
                for i in range(ux.shape[0])
            ]
        )
        u = normalize_data(u)
        # v = np.load('../data/boussinesq_v.npy', allow_pickle=True)[:-1, ...]

    elif dataset_name == "SST":
        u = np.load("../data/sea_surface_noaa.npy", allow_pickle=True)[:2000, ...][
            :, 10:-10, 20:-20
        ]
        u = normalize_data(u)

    elif dataset_name == "2d_cylinder_CFD":
        u_comp = np.load("../data/Vort100.npz", allow_pickle=True)
        # u_comp = np.load('../data/Velocity160.npz', allow_pickle=True)

        u_flat = u_comp["arr_0"]
        print("shape u_flat= " + str(u_flat.shape))
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1)).astype(np.float32)
        u = normalize_data(u)

    elif dataset_name == "2d_sq_cyl":
        u_flat = np.load("../data/sq_cyl_vort.npy", allow_pickle=True)  # sq_cyl_vel
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1)).astype(np.float32)[
            :2000, ...
        ]  # temporarily reducing dataset size
        u = normalize_data(u)

    elif dataset_name == "channel_flow":
        u = np.load("../data/channel_data_2500.npy", allow_pickle=True).astype(
            np.float32
        )
        u = normalize_data(u)

    elif dataset_name == "2d_airfoil":
        u_flat = np.load("../data/airfoil80x320_data.npy", allow_pickle=True)
        print(u_flat.shape)
        u = u_flat.reshape(u_flat.shape[0], 320, 80)
        u = np.transpose(u, (0, 2, 1))[:, :, 140:-20].astype(np.float32)
        u = normalize_data(u)

    elif dataset_name == "2d_plate":
        u_flat = np.load("../data/platekepsilon.npy", allow_pickle=True)
        print(u_flat.shape)
        u = u_flat.reshape(u_flat.shape[0], 360, 180)
        u = np.transpose(u, (0, 2, 1))[:, :-20, :-40].astype(np.float32)
        u = normalize_data(u)

    else:
        print("Dataset Not Found")

    # NOTE 1: Flatten the input data set into a vector
    print(f"Data Loaded in Dataset: {dataset_name} with shape {u.shape[0]}")

    # train/val split
    # NOTE 2: divide training-test to 75% Training 25% test
    # TODO improve this by using train-validation-test cross validtion technique
    train_to_val = 0.75
    # rand_array = np.random.permutation(1500)
    # print(rand_array)

    u_train = u[: int(train_to_val * u.shape[0]), ...]
    u_validation = u[int(train_to_val * u.shape[0]) :, ...]

    print("Training Set Shape= " + str(u_train.shape))
    print("Validation Set Shape= " + str(u_validation.shape))

    # u = insert_time_channel(u, 10)
    # print(u.shape);

    img_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ]
    )

    if transfer_learning:
        print("Using Transfer Learning")
        # final_model = LSTM()
        # pretrained = autoencoder()
        # PATH = "../weights/1000.pth"
        # # PATH = "../weights/bous_500.pth"
        # # pdb.set_trace()
        pre_dataset_name = "2d_cylinder_CFD"
        final_dataset_name = dataset_name
        final_model = UNet_3D(name=final_dataset_name)
        pretrained = UNet_3D(name=pre_dataset_name)

        PATH = f"../results/{pre_dataset_name}/weights/{last_pth}.pth"

        model = load_transfer_learning_UNet_3D(
            pretrained, final_model, PATH, req_grad=False
        )
    else:
        print("Training from scratch UNet3d..")
        model = UNet_3D(name=dataset_name)

    model = model.to(device)

    if args.training:
        # Train data_loader
        train_dataset = AE_3D_Dataset(u_train, dataset_name, transform=img_transform)
        train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=16)
        train_loader = data.DataLoader(train_dataset, **train_loader_args)

        # print(len(train_loader))

        # val data_loader
        validation_dataset = AE_3D_Dataset(
            u_validation, dataset_name, transform=img_transform
        )
        val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
        val_loader = data.DataLoader(validation_dataset, **val_loader_args)

        # Instances of optimizer, criterion, scheduler

        optimizer = optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=False,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=5,
            min_lr=1e-5,
            eps=1e-08,
        )

        # model.load_state_dict(torch.load(Path))
        # print(optimizer)

        Train_Loss = []
        Dev_Loss = []

        Val_loss = {}
        Train_loss = {}
        stack_size = 1
        validation_freq = 20
        # Epoch loop
        for epoch in range(num_epochs + 1):
            start_time = time.time()
            print("Epoch no: ", epoch)
            train_loss = training(model, train_loader, criterion, optimizer)

            if epoch % validation_freq == 0:  # and epoch !=0:
                val_loss = validation(model, val_loader, criterion)
                Val_loss[epoch] = val_loss
                Dev_Loss.append(val_loss)

                Train_Loss.append(train_loss)
                Train_loss[epoch] = train_loss

            if epoch % stack_size == 0:  # and epoch != 0:
                path = f"../results/{dataset_name}/weights/{epoch}.pth"
                torch.save(model.state_dict(), path)
                print(optimizer)

            scheduler.step(train_loss)
            print("Time : ", time.time() - start_time)
            print("=" * 100)
            print()

        # Saving Loss values as dictionaries for later analyses
        save_loss(Train_loss, dataset_name, "train")
        save_loss(Val_loss, dataset_name, "val")
        plot_training(Train_Loss, Dev_Loss)

    if args.testing:
        for pth in range(101):
            test_epoch = pth
            PATH = find_weight(dataset_name, test_epoch)

            print(PATH)

            model.load_state_dict(torch.load(PATH))

            test_dataset = AE_3D_Dataset(
                u_validation, dataset_name, transform=img_transform
            )
            test_loader_args = dict(batch_size=1, shuffle=False, num_workers=16)
            test_loader = data.DataLoader(test_dataset, **test_loader_args)

            labels, preds = test(model, test_loader)
            name = f"../results/{dataset_name}/labels.npy"
            np.save(name, labels)

            name = f"../results/{dataset_name}/predictions.npy"
            np.save(name, preds)

            p1_labels =  []
            p2_labels =  []
            p1_preds =  []
            p2_preds =  []

            for img in range(labels.shape[0]):
                U, S, Vt = np.linalg.svd(labels[img])
                idx = np.argsort(S)[::-1]
                S = S[idx]
                U = U[:, idx]
                for i in range(U.shape[0]):
                    p1_labels.append( U[i][0] * S[0] )
                    p2_labels.append( U[i][1] * S[1] )
            for img in range(preds.shape[0]):
                U, S, Vt = np.linalg.svd(preds[img])
                idx = np.argsort(S)[::-1]
                S = S[idx]
                U = U[:, idx]
                for i in range(U.shape[0]):
                    p1_preds.append( U[i][0] * S[0] )
                    p2_preds.append( U[i][1] * S[1] )

            plt.plot(p1_labels, p2_labels, '-.', linewidth=0.1, color='grey', alpha = 0.7, label='Groundtruth')
            plt.plot(p1_preds, p2_preds, '-.', linewidth=0.1, color='red', alpha=0.7, label='Prediction')
            plt.legend(loc='upper right')
            plt.xlabel('P1')
            plt.ylabel('P2')
            plt.title('PCA of Temporal Image Data')
            plt.savefig(
                    f"../results/{dataset_name}/p1p2_plots/p1p2_{pth}.png",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0)
            plt.close()

            # MSE(dataset_name, preds, labels)

        frames = []
        common_shape = (3351, 2597) # hardcoded
        for pth in range(101):
            img = imageio.imread(f'../results/{dataset_name}/p1p2_plots/p1p2_{pth}.png')
            img = cv2.resize(img, common_shape)
            print("pth: " + str(pth)+ str(img.shape))
            frames.append(img)
        imageio.mimsave(f'../results/{dataset_name}/p1p2_movie.gif', frames, 'GIF', duration=0.1)
        ##### MSE over timesteps
        # errors = [(pred - label) ** 2 for pred, label in zip(preds, labels)]
        # mean_squared_errors = [np.mean(error) for error in errors]

        # # Plot
        # plt.plot(mean_squared_errors, label='Error over time')
        # plt.xlabel('Time steps')
        # plt.ylabel('Mean Squared Error')
        # plt.title('Error between predictions and ground truth over time')
        # plt.legend()
        # plt.savefig(
        #         f"../results/{dataset_name}/mse.png",
        #         dpi=600,
        #         bbox_inches="tight",
        #         pad_inches=0)
        # plt.close()

        # Plot cumlative error
        # Calculate error at each time step (mean squared error in this example)
        # errors = [(pred - label) ** 2 for pred, label in zip(preds, labels)]
        # mean_squared_errors = [np.mean(error) for error in errors]

        # # Calculate the cumulative sum of errors
        # cumulative_errors = np.cumsum(mean_squared_errors)
        # plt.plot(cumulative_errors, label='Cumulative Error')
        # plt.xlabel('Time steps')
        # plt.ylabel('Cumulative Mean Squared Error')
        # plt.title('Cumulative Error over Time')
        
        
        ######## PCA diffs
        # p1_diff = []
        # p2_diff = []
        # # Loop through each timestep
        # for img in range(labels.shape[0]):
        #     # Calculate principal components for labels and predictions
        #     p1_labels, p2_labels = principal_components_p1p2(labels[img])
        #     p1_preds, p2_preds = principal_components_p1p2(preds[img])
            
        #     # Calculate the difference in principal components and append to list
        #     p1_diff.append(p1_labels - p1_preds)
        #     p2_diff.append(p2_labels - p2_preds)

        # # Convert lists to arrays
        # p1_diff = np.array(p1_diff).flatten()
        # p2_diff = np.array(p2_diff).flatten()

        # Flatten the images
        # labels_flat = labels.reshape(labels.shape[0], -1)
        # preds_flat = preds.reshape(preds.shape[0], -1)

        # # Center the data by subtracting the mean
        # labels_centered = labels_flat - np.mean(labels_flat, axis=0)
        # preds_centered = preds_flat - np.mean(preds_flat, axis=0)

        # # Perform SVD
        # U_labels, S_labels, Vt_labels = np.linalg.svd(labels_centered, full_matrices=False)
        # U_preds, S_preds, Vt_preds = np.linalg.svd(preds_centered, full_matrices=False)

        # # Calculate the principal components projections
        # p1_labels = U_labels[:, 0] * S_labels[0]
        # p2_labels = U_labels[:, 1] * S_labels[1]

        # p1_preds = U_preds[:, 0] * S_preds[0]
        # p2_preds = U_preds[:, 1] * S_preds[1]

        # # Calculate the differences in the projections
        # p1_diff = p1_preds - p1_labels
        # p2_diff = p2_preds - p2_labels


        # # Plot the differences
        # plt.scatter(p1_diff, p2_diff, alpha=0.5, s=5)
        # # plt.plot(p1_diff, p2_diff, '-.', linewidth=0.1, color='red', alpha = 0.7)
        # plt.xlabel('Difference in First Principal Component')
        # plt.ylabel('Difference in Second Principal Component')
        # plt.title('Difference in Principal Components: Ground Truth vs Predictions')
        # plt.grid(True)
        # plt.savefig(
        #         f"../results/{dataset_name}/p1_p2_diff.png",
        #         dpi=600,
        #         bbox_inches="tight",
        #         pad_inches=0)
        # plt.close()

        # plotPCAAllScenarios()

        ###


    if args.simulate:
        PATH = find_weight(dataset_name, test_epoch)

        print(PATH)

        model.load_state_dict(torch.load(PATH))

        labels, preds, mse = simulate(model, u_validation, img_transform)
        name = f"../simulate/{dataset_name}/labels.npy"
        np.save(name, labels)

        name = f"../simulate/{dataset_name}/predictions.npy"
        np.save(name, preds)

        name = f"../simulate/{dataset_name}/mse.npy"
        np.save(name, mse)
