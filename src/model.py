import sys; sys.path.append('.')
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from src.utils import cal_accu


class MLP(LightningModule):
    def __init__(self, 
        in_dim: int = 8, 
        out_dim: int = 1,
        hdim: int = 16, 
        lr: float=1e-3):
        super(MLP, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, out_dim)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_nb): 
        x, y = batch
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        y_ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Training Loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb): 
        '''
        Return:
            y <torch.tensor> target
            y_ <torch.tensor> prediction
        '''
        x, y = batch
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        y_ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Validation Loss', loss, prog_bar=True, logger=True)
        return y, y_

    def validation_epoch_end(self, validation_step_outputs):
        tar_list = []
        out_list = []
        for (tar, out) in validation_step_outputs:
            tar_list.append(tar.detach().cpu())
            out_list.append(out.detach().cpu())
        tar = torch.stack(tar_list[:-1], dim=1)
        tar = tar.view(-1)
        out = torch.stack(out_list[:-1], dim=1)
        out = torch.squeeze(out).view(-1)
        rmse, r2, corr = cal_accu(tar, out)
        self.log('Validation RMSE', rmse)
        self.log('Validation R2', r2)
        self.log('Validation Corr', corr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LSTM(LightningModule):
    def __init__(self, 
        in_dim: int = 8, 
        hdim: int = 16, 
        num_layers: int = 1,
        lr: float=1e-3):
        super(MLP, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.LSTM(in_dim, hdim, num_layers=num_layers),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Training Loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Validation Loss', loss, prog_bar=True, logger=True)
        return y, y_

    def validation_epoch_end(self, validation_step_outputs):
        tar_list = []
        out_list = []
        for (tar, out) in validation_step_outputs:
            tar_list.append(tar.detach().cpu())
            out_list.append(out.detach().cpu())
        tar = torch.stack(tar_list[:-1], dim=1)
        tar = tar.view(-1)
        out = torch.stack(out_list[:-1], dim=1)
        out = torch.squeeze(out).view(-1)
        rmse, r2, corr = cal_accu(tar, out)
        self.log('Validation RMSE', rmse)
        self.log('Validation R2', r2)
        self.log('Validation Corr', corr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class GRU(LightningModule):
    def __init__(self, 
        in_dim: int = 8, 
        hdim: int = 16, 
        num_layers: int = 1,
        lr: float=1e-3):
        super(MLP, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            nn.GRU(in_dim, hdim, num_layers=num_layers),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Training Loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Validation Loss', loss, prog_bar=True, logger=True)
        return y, y_

    def validation_epoch_end(self, validation_step_outputs):
        tar_list = []
        out_list = []
        for (tar, out) in validation_step_outputs:
            tar_list.append(tar.detach().cpu())
            out_list.append(out.detach().cpu())
        tar = torch.stack(tar_list[:-1], dim=1)
        tar = tar.view(-1)
        out = torch.stack(out_list[:-1], dim=1)
        out = torch.squeeze(out).view(-1)
        rmse, r2, corr = cal_accu(tar, out)
        self.log('Validation RMSE', rmse)
        self.log('Validation R2', r2)
        self.log('Validation Corr', corr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class Our(LightningModule):
    def __init__(self, 
        n_features: int = 9,
        window_size: int = 200, 
        out_dim: int = 2, 
        kernel_size: int = 5,
        gru_n_layers: int = 1, 
        gru_hid_dim: int = 64, 
        forecaster_n_layers: int = 6,
        forecaster_hid_dim: int = 256,
        recon_n_layers: int = 6,
        recon_hid_dim: int = 256,
        dropout: float = 0.3,
        alpha: float = 0.2,
        lr: float=1e-3):
        super(MLP, self).__init__()
        self.lr = lr
        self.layers = nn.Sequential(
            MTAD_GAT(
                n_features=n_features,
                window_size=window_size,
                out_dim=out_dim,
                kernel_size=kernel_size,
                use_gatv2=True, 
                feat_gat_embed_dim=None,
                time_gat_embed_dim=None,
                gru_n_layers=gru_n_layers,
                gru_hid_dim=gru_hid_dim,
                forecast_n_layers=forecaster_n_layers,
                forecast_hid_dim=forecaster_hid_dim, 
                recon_n_layers=recon_n_layers,
                recon_hid_dim=recon_hid_dim, 
                dropout=dropout,
                alpha=alpha
            )
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Training Loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb): 
        x, y, _ = batch
        x = torch.permute(x.to(torch.float32), (1, 0, 2))
        y = y[:, :, -1].to(torch.float32)
        y_, _ = self(x)
        loss = nn.MSELoss()(y, y_)
        self.log('Validation Loss', loss, prog_bar=True, logger=True)
        return y, y_

    def validation_epoch_end(self, validation_step_outputs):
        tar_list = []
        out_list = []
        for (tar, out) in validation_step_outputs:
            tar_list.append(tar.detach().cpu())
            out_list.append(out.detach().cpu())
        tar = torch.stack(tar_list[:-1], dim=1)
        tar = tar.view(-1)
        out = torch.stack(out_list[:-1], dim=1)
        out = torch.squeeze(out).view(-1)
        rmse, r2, corr = cal_accu(tar, out)
        self.log('Validation RMSE', rmse)
        self.log('Validation R2', r2)
        self.log('Validation Corr', corr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons


class MTAD_GAT_Analysis(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(MTAD_GAT_Analysis, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
