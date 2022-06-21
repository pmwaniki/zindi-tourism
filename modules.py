import torch
import torch.nn as nn

def mlp_block(dim_hidden,dropout,n_hidden=1):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU()
        )
    for _ in range(n_hidden)])

class Classifier(nn.Module):
    def __init__(self,dim_x,cat_idx,cat_dims,emb_size=8,n_hidden=2,dim_hidden=32,dropout=0.01,n_out=6):
        super().__init__()
        self.cat_idx=cat_idx
        self.dim_x=dim_x
        fc_input_dim=int(len(cat_idx)*emb_size+(dim_x-len(cat_idx)))
        self.emb_layers=nn.ModuleList()
        for emb_dim in cat_dims:
            self.emb_layers.append(nn.Embedding(num_embeddings=emb_dim,embedding_dim=emb_size))

        self.first_layer=nn.Sequential(
            nn.Linear(fc_input_dim,dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU()
        )
        self.mid_layers=mlp_block(dim_hidden,dropout,n_hidden=n_hidden)
        self.fc=nn.Linear(dim_hidden,n_out)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,)
                torch.nn.init.constant_(m.bias,0)


    def forward(self,x):
        embeddings_list=[]
        for i,col in enumerate(self.cat_idx):
            embeddings_list.append(self.emb_layers[i](x[:,col].type(torch.int64)))
        embeddings=torch.cat(embeddings_list,dim=1)
        non_embeddings=x[:,[i for i in range(self.dim_x) if i not in self.cat_idx]]
        out=torch.cat([embeddings,non_embeddings],dim=1)
        out=self.first_layer(out)
        out=self.mid_layers(out)
        out=self.fc(out)
        return out


class ClassifierRegularized(nn.Module):
    def __init__(self,dim_x,cat_idx,cat_dims,emb_size=8,n_hidden=2,#n_hidden_decoder=2,
                 dim_hidden=32,dropout=0.01,n_out=6,dim_z=8):
        super(ClassifierRegularized, self).__init__()
        self.encoder=Classifier(dim_x,cat_idx,cat_dims,emb_size,n_hidden=n_hidden,
                                dim_hidden=dim_hidden,dropout=dropout,n_out=dim_z)
        self.decoder_mask=nn.Sequential(
            nn.Linear(dim_z,dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden,dim_x)
        )
        self.decoder_outcome=nn.Sequential(
            nn.Linear(dim_z,dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden,n_out)
        )

        for m in self.modules():
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,)
                torch.nn.init.constant_(m.bias,0)

    def forward(self,x):
        z=self.encoder(x)
        mask=self.decoder_mask(z)
        out=self.decoder_outcome(z)
        return mask,out
