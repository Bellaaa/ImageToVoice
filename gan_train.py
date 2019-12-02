import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from torchsummary import summary

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network
from utils import Meter, cycle, save_model, save_model_state_dict
from loss import *
from dataset import reload_batch_face, reload_batch_voice

# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num, voice_dict, face_dict = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num

print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)  # voice embedding
# g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)
f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
g_net, g_optimizer = get_network('u', NETWORKS_PARAMETERS, train=True)  # unet
d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)  # discriminator
c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)  # classifier, train=False

# d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.96)
# g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.96)
# summary(g_net, input_size=(3, 64, 64))


# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')

meter_D_real = Meter('D_real', 'avg', ':3.2f')
meter_D_fake = Meter('D_fake', 'avg', ':3.2f')
meter_C_real = Meter('C_real', 'avg', ':3.2f')
meter_GD_fake = Meter('G_D_fake', 'avg', ':3.2f')
meter_GC_fake = Meter('G_C_fake', 'avg', ':3.2f')
meter_G_L2_fake = Meter('G_l2_fake', 'avg', ':3.2f')
""" """

print('Training models...')
min_g_loss = None
min_d_loss = None
str_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
f_log = 'training_log_{}'.format(str_timestamp)

for it in range(DATASET_PARAMETERS['num_batches']):
    # data
    start_time = time.time()

    voiceB, voiceB_label = next(voice_iterator)
    faceA, faceA_label = next(face_iterator)  # real face

    # TODO: since voiceB and faceA in different identities,
    #  need to reuse load_voice and load_face to get corresponding faceB and voiceA
    faceB_items = [face_dict[v_label.item()] for v_label in voiceB_label]
    voiceA_items = [voice_dict[f_label.item()] for f_label in faceA_label]
    faceB = reload_batch_face(faceB_items)
    voiceA = reload_batch_voice(voiceA_items, DATASET_PARAMETERS['nframe_range'][1])
    # noise = 0.05 * torch.randn(DATASET_PARAMETERS['batch_size'], 64, 1, 1)  # shape 4d!

    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']:
        voiceB, voiceB_label = voiceB.cuda(), voiceB_label.cuda()
        faceA, faceA_label = faceA.cuda(), faceA_label.cuda()
        faceB, voiceA = faceB.cuda(), voiceA.cuda()
        # real_label, fake_label = real_label.cuda(), fake_label.cuda()
        # noise = noise.cuda()
    data_time.update(time.time() - start_time)

    # TODO: scale the input images, notice when inference ??
    # scaled_images = face * 2 - 1

    # get voice embeddings
    embedding_B = e_net(voiceB)
    embedding_B = F.normalize(embedding_B).view(embedding_B.size()[0], -1)
    # introduce some permutations to voice --> deprecated
    # embeddings = embeddings + noise
    # embeddings = F.normalize(embeddings)

    # 0. get generated faces
    fake_faceB = g_net(faceA, embedding_B)
    # TODO: introduce some permutations to image !!!

    # ============================================
    #            TRAIN THE DISCRIMINATOR
    # ============================================

    # if it != 1 and it % 10 == 1:
    f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()

    # 1. Train with real images
    D_real_A = d_net(f_net(faceA))
    D_real_A_loss = true_D_loss(torch.sigmoid(D_real_A))

    # 2. Train with fake images
    D_fake_B = d_net(f_net(fake_faceB).detach())
    # D_fake = d_net(f_net(fake_face.detach()))  # TODO: is detach necessary here ???
    D_fake_B_loss = fake_D_loss(torch.sigmoid(D_fake_B))

    # 3. Train with identity / gender classification
    real_classification = c_net(f_net(faceA))
    C_real_loss = identity_D_loss(F.log_softmax(real_classification, dim=1), faceA_label)

    # D_real_loss = F.binary_cross_entropy(torch.sigmoid(D_real), real_label)
    # D_fake_loss = F.binary_cross_entropy(torch.sigmoid(D_fake), fake_label)

    # update meters
    meter_D_real.update(D_real_A_loss.item())
    meter_D_fake.update(D_fake_B_loss.item())
    meter_C_real.update(C_real_loss.item())

    # backprop
    D_loss = D_real_A_loss + D_fake_B_loss + C_real_loss
    D_loss.backward()
    f_optimizer.step()
    c_optimizer.step()
    # if it % 10 == 0:
    d_optimizer.step()
    # d_optimizer.zero_grad()

    # =========================================
    #            TRAIN THE GENERATOR
    # =========================================
    g_optimizer.zero_grad()

    # 0. get generated faces
    fake_faceB = g_net(faceA, embedding_B)

    # 1. Train with discriminator
    D_fake_B = d_net(f_net(fake_faceB))
    D_B_loss = true_D_loss(torch.sigmoid(D_fake_B))

    # 2. Train with classifier
    fake_classfication = c_net(f_net(fake_faceB))
    C_fake_loss = identity_D_loss(F.log_softmax(fake_classfication, dim=1), voiceB_label)
    # C_fake_loss = F.nll_loss(F.log_softmax(fake_classfication, 1), voice_label)

    # GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(D_fake), real_label)
    # GC_fake_loss = F.nll_loss(F.log_softmax(fake_classfication, 1), voice_label)

    # 3. Train with L2 loss
    l2loss = l2_loss_G(fake_faceB, faceB)

    # 4. Train with consistency loss
    # TODO: to be tested, after getting embedding_A and ??
    # scaled_fake = fake_face * 2 - 1
    # get voice embeddings
    embedding_A = e_net(voiceA)
    embedding_A = F.normalize(embedding_A).view(embedding_A.size()[0], -1)
    fake_faceA = g_net(fake_faceB, embedding_A)
    consistency_loss = l2_loss_G(fake_faceA, faceA)

    # backprop
    G_loss = D_B_loss + C_fake_loss + l2loss + consistency_loss
    G_loss.backward()
    meter_GD_fake.update(D_B_loss.item())
    meter_GC_fake.update(C_fake_loss.item())
    meter_G_L2_fake.update(l2loss.item() + consistency_loss.item())
    g_optimizer.step()

    batch_time.update(time.time() - start_time)

    # print status
    if it % DATASET_PARAMETERS['print_stat_freq'] == 0:
        str_log = str(iteration) + str(data_time) + str(batch_time) + str(meter_D_real) + \
                  str(meter_D_fake) + str(meter_C_real) + str(meter_GD_fake) + str(meter_GC_fake)
        # print(iteration, data_time, batch_time,
        #       meter_D_real, meter_D_fake, meter_C_real, meter_GD_fake, meter_GC_fake)
        print(str_log)
        with open(f_log, 'a+') as f:
            f.write(str_log)

        data_time.reset()
        batch_time.reset()
        meter_D_real.reset()
        meter_D_fake.reset()
        meter_C_real.reset()
        meter_GD_fake.reset()
        meter_GC_fake.reset()

    # save intermediate models for visualization purpose
    if it % DATASET_PARAMETERS['save_freq'] == 0:
        f_cur_model = '_iter{}.pkl'.format(it)
        f_cur_model = NETWORKS_PARAMETERS['u']['model_path'].split('.')[0] + f_cur_model
        save_model_state_dict(g_net, f_cur_model)

    del voiceB_label
    del voiceB
    del faceA_label
    del faceA
    del voiceA
    del faceB

    # save the best model
    """ """
    if min_g_loss is None or G_loss < min_g_loss:
        min_g_loss = G_loss
        save_model(g_net, NETWORKS_PARAMETERS['u']['model_path'])
    if min_d_loss is None or D_loss < min_d_loss:
        min_d_loss = D_loss
        save_model(d_net, NETWORKS_PARAMETERS['d']['model_path'])

    iteration.update(1)

