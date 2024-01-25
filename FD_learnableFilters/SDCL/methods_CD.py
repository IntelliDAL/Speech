import torch
from untils import Dataloader
from torch.autograd import Variable


def train(args, data, model_training, Discriminator):
    # Loss and optimizer
    optimizer_D = torch.optim.RMSprop((param for param in Discriminator.parameters() if param.requires_grad), lr=args.lr_D, alpha=0.95, eps=1e-8, weight_decay=0.001)
    optimizer = torch.optim.Adam(model_training.parameters(), lr=args.learning_rate, weight_decay=0.001)

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.95)
    optim_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_D, step_size=25, gamma=0.95)

    # 模型读取数据后，先进性傅立叶变换。然后进行数据增强X and X'
    train_dataset = Dataloader(args, data)

    # Train the models
    for epoch in range(args.num_epochs):
        model_training.train()
        Discriminator.train()
        total_step = len(train_dataset)
        for batch_idx, batch_data in enumerate(train_dataset):
            gaussian = torch.rand((args.batch_size, args.filter_num, args.frame_num)).cuda()
            batch_data = batch_data.transpose(1, 0)
            batchSamples, batchSamples_ = batch_data[0], batch_data[1]
            ###############################################
            # Train CD
            ###############################################
            for iters in range(2):
                optimizer_D.zero_grad()
                data_dict = model_training.forward(batchSamples.type(torch.FloatTensor).cuda(), batchSamples_.type(torch.FloatTensor).cuda())
                I_online, I_target = data_dict['I_online'], data_dict['I_target']

                # 上下分支的共性特征 共性特征和高斯分布之间的距离比较进
                # fake
                # 共性特征为fake
                code_IO = Discriminator(I_online)
                code_IT = Discriminator(I_target)
                # 差异性特征 距离相对高斯分布较远, 差异性为True  高斯分布为fake,便会越来越远.
                # real
                real_validity = Discriminator(gaussian)
                # -----------------------------
                gradient_penalty_IO = calc_gradient_penalty(Discriminator, gaussian, I_online)
                gradient_penalty_IT = calc_gradient_penalty(Discriminator, gaussian, I_target)

                # Loss_IO = (torch.mean(code_IO) - torch.mean(real_validity))
                # Loss_IT = (torch.mean(code_IT) - torch.mean(real_validity))
                Loss_IO = (torch.mean(code_IO) - torch.mean(real_validity)) + gradient_penalty_IO
                Loss_IT = (torch.mean(code_IT) - torch.mean(real_validity)) + gradient_penalty_IT
                # ------------------------------------------
                Loss_CD = (Loss_IO + Loss_IT)/2
                Loss_CD.backward()
                optimizer_D.step()
                optim_scheduler_D.step()
            ###############################################
            # Train Encoder
            """将对比学习的整体模块看作Encoder I整体
            每次loss 更新后，单独更几次Discriminator
            """
            ###############################################
            data_dict = model_training.forward(batchSamples.type(torch.FloatTensor).cuda(), batchSamples_.type(torch.FloatTensor).cuda())
            Loss_back = data_dict['loss'].mean()
            """计算分布约束的loss"""
            # ------------------------------------------
            I_online, I_target = data_dict['I_online'], data_dict['I_target']
            """2个Encoder的输出变量输入到判别器当中进行分布计算"""
            # 生成器中, loss是real - fake， real是常数
            Loss_IO = -Discriminator(I_online).mean()
            Loss_IT = -Discriminator(I_target).mean()

            Loss_Encoder = Loss_back + (Loss_IT + Loss_IO)/2
            model_training.zero_grad()
            Loss_Encoder.backward()
            optimizer.step()
            optim_scheduler.step()
            print('Epoch [{}/{}], Step [{}/{}, Loss_E {:.4f}, Loss_D {:.4f}]'.format(epoch, args.num_epochs, batch_idx, total_step, Loss_Encoder, Loss_CD))
            if epoch % args.log_step == 0:
                print('Learning Rate:', optimizer.param_groups[0]['lr'])
    return model_training



LAMBDA = 10
def calc_gradient_penalty(netD, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size() == x_gen.size(), "real and sampled sizes do not match"
    BATCH_SIZE, Filter, frame = x.shape
    alpha = torch.rand(BATCH_SIZE, 1, 1).repeat(1, Filter, frame).cuda()
    interpolates = alpha * x.data + (1 - alpha) * x_gen.data
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True, retain_graph=True, allow_unused=True)[0]
    gradients = Variable(gradients, requires_grad=True)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.data.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


