import torch
import torch.nn as nn





class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        # return (self.H, self.C), self.H  # (hidden, output)
        return self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                      stride=stride, padding=1),
            nn.GroupNorm(4, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                               stride=stride, padding=1, output_padding=output_padding),
            nn.GroupNorm(4, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class image_encoder(nn.Module):
    def __init__(self, nc, device):
        super(image_encoder, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf//2, stride=1)  # (nf) x 896 x 896
        self.c2 = dcgan_conv(nf//2, nf * 1, stride=2)  # (2*nf) x 448 x 448
        self.c3 = dcgan_conv(nf * 1, nf * 2, stride=2)  # (4*nf) x 224 x 224
        self.c4 = dcgan_conv(nf * 2, nf * 4, stride=2)  # (8*nf) x 8 x 8
        self.c5 = dcgan_conv(nf * 4, nf * 8, stride=2)  # (8*nf) x 8 x 8
        self.c6 = dcgan_conv(nf * 8, nf * 16, stride=2)  # (8*nf) x 8 x 8

        self.convlstm_5 = ConvLSTM(input_shape=(56, 56), input_dim=nf * 8, hidden_dims=[nf * 8], kernel_size=kernel_size,
                                   device=device).to(device)
        self.convlstm_6 = ConvLSTM(input_shape=(28, 28), input_dim=nf * 16, hidden_dims=[nf * 16], kernel_size=kernel_size,
                                   device=device).to(device)

    def forward(self, input, first_timestep):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)

        h51 = self.c5(h4)
        h5 = self.convlstm_5(h51, first_timestep)[-1]  # (nf*16) x 8 x 8

        h61 = self.c6(h5)
        h6 = self.convlstm_6(h61, first_timestep)[-1]  # (nf*16) x 8 x 8

        return [h1, h2, h3, h4, h5, h6]


class image_decoder(nn.Module):
    def __init__(self, nc, device):
        super(image_decoder, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        self.upc1 = dcgan_upconv(nf * 16, nf * 8, stride=2)  # (nf*2) x 16 x 16
        self.upc2 = dcgan_upconv(nf * 8, nf * 4, stride=2)  # (nf*2) x 16 x 16
        self.upc3 = dcgan_upconv(nf * 4, nf * 2, stride=2)  # (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf * 1, stride=2)  # (nf) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 1, nf // 2, stride=2)  # (nf) x 64 x 64
        self.upc6 = nn.ConvTranspose2d(nf // 2, nc, kernel_size=(3, 3), stride=1, padding=1)  # (nc) x 64 x 64

        self.convlstm_5 = ConvLSTM(input_shape=(56, 56), input_dim=nf * 8 * 2, hidden_dims=[nf * 8], kernel_size=kernel_size,
                                   device=device).to(device)
        self.convlstm_6 = ConvLSTM(input_shape=(28, 28), input_dim=nf * 16, hidden_dims=[nf * 16], kernel_size=kernel_size,
                                   device=device).to(device)

    def forward(self, input, first_timestep):
        output, skip = input  # output: (4*nf) x 16 x 16
        output_6, output_5, output_4, output_3 = output

        d1 = self.convlstm_6(torch.cat([output_6], dim=1), first_timestep)[-1]  # (nf*16) x 8 x 8
        d21 = self.upc1(d1)                         # (nf*8) x 16 x 16
        d2 = self.convlstm_5(torch.cat([d21, output_5], dim=1), first_timestep)[-1]  # (nf*8) x 16 x 16
        d31 = self.upc2(d2)     # (nf*4) x 32 x 32
        d41 = self.upc3(d31)     # (nf*1) x 64 x 64
        d5 = self.upc4(d41)  # (nf*1) x 64 x 64
        d6 = self.upc5(d5)  # (nf*1) x 64 x 64
        d7 = self.upc6(d6)  # (nf*1) x 64 x 64

        return d7


class EncoderRNN(torch.nn.Module):
    def __init__(self, device):
        super(EncoderRNN, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        self.image_cnn_enc = image_encoder(1, device).to(device)  # image encoder 64x64x1 -> 16x16x64
        self.image_cnn_dec = image_decoder(1, device).to(device)  # image decoder 16x16x64 -> 64x64x1

        self.convlstm_5 = ConvLSTM(input_shape=(56, 56), input_dim=nf * 8, hidden_dims=[nf * 8],
                                   kernel_size=kernel_size, device=device).to(device)
        self.convlstm_6 = ConvLSTM(input_shape=(28, 28), input_dim=nf * 16, hidden_dims=[nf * 16],
                                   kernel_size=kernel_size, device=device).to(device)

    def forward(self, input, first_timestep=False):
        skip = self.image_cnn_enc(input, first_timestep)
        [h1, h2, h3, h4, h5, h6] = skip
        output_6 = self.convlstm_6(h6, first_timestep)
        output_5 = self.convlstm_5(h5, first_timestep)

        output = [output_6[-1], output_5[-1], 0, 0]

        output_image = torch.sigmoid(self.image_cnn_dec([output, skip], first_timestep))

        return output_image