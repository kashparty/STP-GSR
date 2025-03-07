# import gc
# import torch
# import tempfile
# import numpy as np
# from tqdm import tqdm
# from torch_geometric.data import Data

# from src.models.stp_gsr import STPGSR
# from src.models.direct_sr import DirectSR
# from src.plot_utils import (
#     plot_grad_flow, 
#     plot_adj_matrices, 
#     create_gif_grad, 
#     create_gif_adj,
#     plot_losses,
# )
# from src.dual_graph_utils import revert_dual


# def load_model(config):
#     if config.model.name == 'stp_gsr':
#         return STPGSR(config)
#     elif config.model.name == 'direct_sr':
#         return DirectSR(config)
#     else:
#         raise ValueError(f"Unsupported model type: {config.model.name}")
    

# def eval(config, model, source_data, target_data, critereon):
#     n_target_nodes = config.dataset.n_target_nodes  # n_t
    
#     model.eval()

#     eval_output = []

#     eval_loss = []

#     with torch.no_grad():
#         for source, target in zip(source_data, target_data):
#             source_g = source['pyg']    
#             target_m = target['mat']    # (n_t, n_t)

#             model_pred, model_target = model(source_g, target_m) 

#             if config.model.name == 'stp_gsr':
#                 pred_m = revert_dual(model_pred, n_target_nodes)    # (n_t, n_t)
#                 pred_m = pred_m.cpu().numpy()
#             else:
#                 pred_m = model_pred.cpu().numpy()

#             eval_output.append(pred_m)

#             t_loss = critereon(model_pred, model_target)

#             eval_loss.append(t_loss) 

#     eval_loss = torch.stack(eval_loss).mean().item()

#     model.train()

#     return eval_output, eval_loss


# def train(config, 
#           source_data_train, 
#           target_data_train, 
#           source_data_val, 
#           target_data_val,
#           res_dir):
#     n_target_nodes = config.dataset.n_target_nodes  # n_t

#     # Initialize model, optmizer, and loss function
#     model = load_model(config)
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     print(model)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.experiment.lr)
#     critereon = torch.nn.L1Loss()

#     train_losses = []
#     val_losses = []
 

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model.train()
#         step_counter = 0

#         for epoch in range(config.experiment.n_epochs):
#             batch_counter = 0
#             epoch_loss = 0.0

#             # Shuffle training data
#             random_idx = torch.randperm(len(source_data_train))
#             source_train = [source_data_train[i] for i in random_idx]
#             target_train = [target_data_train[i] for i in random_idx]

#             # Iteratively train on each sample. 
#             # (Using single sample training and gradient accummulation as the baseline IMANGraphNet model is memory intensive)
#             for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
#                 source_g = source['pyg']
#                 source_m = source['mat']    # (n_s, n_s)
#                 target_m = target['mat']    # (n_t, n_t)

#                 # We pass the target matrix to the forward pass for consistency:
#                 # For our STP-GSR model, its easier to directly compare dual graph features of shape (n_t*(n_t-1)/2, 1)
#                 # Whereas, DirectSR model predicts the target matrix directly of shape (n_t, n_t)
#                 if config.model.name == 'stp_gsr':
#                     model_pred, model_target = model(source_g, target_m)      # both (n_t*(n_t-1)/2, 1)
#                 else:
#                     model_pred, model_target = model(source_g, target_m)

#                 loss = critereon(model_pred, model_target)
#                 loss.backward()

#                 epoch_loss += loss.item()
#                 batch_counter += 1

#                 # Log progress and do mini-batch gradient descent
#                 if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_train):
#                     # Log gradients for this iteration
#                     # plot_grad_flow(model.named_parameters(), step_counter, tmp_dir)


#                     # # Predicetd and target matrices for plotting
#                     # pred_plot = model_pred.detach()
#                     # target_plot = model_target.detach()

#                     # # Convert edge features to adjacency matrices
#                     # if config.model.name == 'stp_gsr':
#                     #     pred_plot = revert_dual(pred_plot, n_target_nodes) # (n_t, n_t)
#                     #     target_plot = revert_dual(target_plot, n_target_nodes) # (n_t, n_t)

#                     # pred_plot_m = pred_plot.cpu().numpy()
#                     # target_plot_m = target_plot.cpu().numpy()

#                     # # Log source, target, and predicted adjacency matrices for this iteration
#                     # plot_adj_matrices(source_m, pred_plot_m, target_plot_m, step_counter, tmp_dir)
                    
#                     # Perform gradient descent
#                     optimizer.step()
#                     optimizer.zero_grad()

#                     step_counter += 1

#                     torch.cuda.empty_cache()
#                     gc.collect()

#             epoch_loss = epoch_loss / len(source_train)
#             print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Train Loss: {epoch_loss}")
#             train_losses.append(epoch_loss)

#             # Log validation loss
#             if config.experiment.log_val_loss:
#                 _, val_loss = eval(config, model, source_data_val, target_data_val, critereon)
#                 print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Val Loss: {val_loss}")
#                 val_losses.append(val_loss)

#         # Save and plot losses
#         np.save(f'{res_dir}/train_losses.npy', np.array(train_losses))
#         np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))
#         plot_losses(train_losses, 'train', res_dir)
#         plot_losses(val_losses, 'val', res_dir)

#         # # Create gif for gradient flows
#         # gif_path = f"{res_dir}/gradient_flow.gif"
#         # create_gif_grad(tmp_dir, gif_path)
#         # print(f"Gradient flow saved as {gif_path}")

#         # # Create gif for training samples
#         # gif_path = f"{res_dir}/train_samples.gif"
#         # create_gif_adj(tmp_dir, gif_path)
#         # print(f"Training samples saved as {gif_path}")

#         # Save model
#         model_path = f"{res_dir}/model.pth"
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved as {model_path}")

#     return {
#         'model': model,
#         'critereon': critereon,
#     }

import gc
import torch
import tempfile
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

from src.models.stp_gsr import STPGSR
from src.models.direct_sr import DirectSR
from src.plot_utils import (
    plot_losses,
)
from src.dual_graph_utils import revert_dual

from src.models.stp_gsr import Discriminator


def load_model(config):
    if config.model.name == 'stp_gsr':
        return STPGSR(config)
    elif config.model.name == 'direct_sr':
        return DirectSR(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model.name}")


def eval(config, model, source_data, target_data, critereon_L1):
    n_target_nodes = config.dataset.n_target_nodes 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    eval_output = []
    eval_loss = []

    with torch.no_grad():
        for source, target in zip(source_data, target_data):
            source_g = source['pyg'].to(device)
            target_m = target['mat'].to(device)

            model_pred, model_target, _, _ = model(source_g, target_m) 
            pred_m = revert_dual(model_pred, n_target_nodes)    # (n_t, n_t)
            pred_m = pred_m.cpu().numpy()
            eval_output.append(pred_m)

            t_loss = critereon_L1(model_pred, model_target)
            eval_loss.append(t_loss)

    eval_loss = torch.stack(eval_loss).mean().item()
    model.train()

    return eval_output, eval_loss


def train(config, 
          source_data_train, 
          target_data_train, 
          source_data_val, 
          target_data_val,
          res_dir):
    n_target_nodes = config.dataset.n_target_nodes  

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(model)

    # Optimizers for Generator (STP-GSR) and Discriminator
    optimizerG = torch.optim.Adam(model.parameters(), lr=config.experiment.lr)
    optimizerD = torch.optim.Adam(model.discriminator.parameters(), lr=config.experiment.lr)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss()
    criterion_BCE = torch.nn.BCELoss()

    train_losses_G = []
    train_losses_D = []
    val_losses = []

    model.train()

    for epoch in range(config.experiment.n_epochs):
        batch_counter = 0
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        random_idx = torch.randperm(len(source_data_train))
        source_train = [source_data_train[i] for i in random_idx]
        target_train = [target_data_train[i] for i in random_idx]

        for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
            source_g = source['pyg'].to(device)
            target_m = target['mat'].to(device)

            # -----------------------------------
            # Train Discriminator
            # -----------------------------------
            for param in model.discriminator.parameters():
                param.requires_grad = True  # Enable gradient computation

            optimizerD.zero_grad()

            with torch.no_grad():  # Freeze Generator (STP-GSR) when training Discriminator
                model_pred, model_target, fake_labels, real_labels = model(source_g, target_m)

            real_loss = criterion_BCE(real_labels, torch.ones_like(real_labels, requires_grad=True))
            fake_loss = criterion_BCE(fake_labels, torch.zeros_like(fake_labels, requires_grad=True))
            disc_loss = (real_loss + fake_loss) / 2  

            disc_loss.backward()
            optimizerD.step()
            epoch_loss_D += disc_loss.item()

            # -----------------------------------
            # Train Generator (STP-GSR)
            # -----------------------------------
            optimizerG.zero_grad()

            # Ensure Discriminator is NOT updated when training Generator
            model_pred, model_target, fake_labels, _ = model(source_g, target_m)

            # Generator should try to fool Discriminator
            l1_loss = criterion_L1(model_pred, model_target)  
            gen_loss = criterion_BCE(fake_labels, torch.ones_like(fake_labels))  

            total_loss = l1_loss + 0.1 * gen_loss  
            total_loss.backward()
            optimizerG.step()

            epoch_loss_G += total_loss.item()
            batch_counter += 1

            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss_G /= len(source_train)
        epoch_loss_D /= len(source_train)

        print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Generator Loss: {epoch_loss_G}, Discriminator Loss: {epoch_loss_D}")
        train_losses_G.append(epoch_loss_G)
        train_losses_D.append(epoch_loss_D)

        # Validation Loss
        if config.experiment.log_val_loss:
            val_loss = eval(config, model, source_data_val, target_data_val)
            print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Val Loss: {val_loss}")
            val_losses.append(val_loss)

    # Save and plot losses
    np.save(f'{res_dir}/train_losses_G.npy', np.array(train_losses_G))
    np.save(f'{res_dir}/train_losses_D.npy', np.array(train_losses_D))
    np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))
    plot_losses(train_losses_G, 'train_G', res_dir)
    plot_losses(train_losses_D, 'train_D', res_dir)
    plot_losses(val_losses, 'val', res_dir)

    # Save model
    model_path = f"{res_dir}/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    return {
        'model': model,
        'criterion_L1': criterion_L1,
        'criterion_BCE': criterion_BCE
    }
