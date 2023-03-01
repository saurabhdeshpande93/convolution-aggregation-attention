from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math
import torch.nn.functional as F
from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverFeatureExtractor, PerceiverModel
import Utils
import os

from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverClassificationDecoder,
    PerceiverBasicDecoder,
    PerceiverProjectionDecoder,
    PerceiverProjectionPostprocessor,
)
import DatasetHandler



class SCCPerceiver(object):

    def __init__(self,in_size):
        self.config = PerceiverConfig()

        self.config.d_latents=210
        self.config.d_model=in_size
        self.config.max_position_embeddings=1024
        self.config.num_latents=128
        self.config.num_blocks=3
        self.config.num_self_attention_heads=2
        self.config.num_cross_attention_heads=2
        self.config.num_self_attends_per_block=2
        self.config.attention_probs_dropout_prob=0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: "+ str(self.device))
        self.batch_sz=1

        self.preprocessor = None
        self.postprocessor = None
        self.decoder = None
        self.model = None

        self.TrainDataset = None
        self.TestDataset = None
        self.CustomDataset = None

        self._setDecoder()
        self._setPreproccesor()
        self._setPostprocessor()
        self._setModel()
        self.criterion = torch.nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.optimizer = AdamW(self.model.parameters(), lr=100e-6)
        self.minloss = 100000

        self.name="perceiver"
        self.state_handler=Utils.model_state_handler(self.name)
        self.it = self.state_handler.it

    def __call__(self, sample):
        print("test")

    def Train(self,epochs, eval_interval):
        n_steps_total=0
        self.model.train()
        for epoch in range(epochs):
            print("Epoch:", epoch)
            n_steps=0
            eval_loss = 0
            train_loss = 0
            for batch_idx, sample in enumerate(self.TrainDataset):

                inputs = sample["features"].to(self.device)
                labels = sample["labels"].to(self.device)


                self.optimizer.zero_grad()
                inputs=inputs.unsqueeze(1)
                # print("inputs_shape= ", inputs.shape)
                outputs = self.model(inputs=inputs.to(self.device))
                # print("last hidden_shape= ", outputs.last_hidden_state.shape)
                logits = outputs.logits.squeeze()
                # print("logits_shape= ",logits.shape)
                # print("labels_shape= ",labels.shape)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()




                n_steps = n_steps + 1
                n_steps_total = n_steps_total + 1
                train_loss = train_loss + loss.item()
            eval_loss=self.Eval()
            print(f"Loss Train: {train_loss/n_steps}, Epoch: {epoch}, Total steps: {n_steps}")
            f = open("loss.txt", 'a')
            f.write(str(epoch)+" "+str(train_loss/n_steps)+" "+str(eval_loss)+'\n')
            f.close()
            if eval_loss < self.minloss:
                self.minloss=eval_loss
                # self.Save(str(eval_loss))
            save_name=self.state_handler.Update(self.it+epoch,train_loss/n_steps,eval_loss)
            if save_name != None:
                self.Save(save_name[1])
                if os.path.exists(save_name[0]+".pt"):
                    os.remove(save_name[0]+".pt")


    def Eval(self,custom=False):
        self.model.eval()

        with torch.no_grad():
            T_loss=0
            n_steps=0
            if custom==False:
                for batch_idx, sample in enumerate(self.TestDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)
                    inputs=inputs.unsqueeze(1)
                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()

                    loss = self.criterionL1(logits, labels)
                    T_loss = T_loss + loss.item()
            else:
                for batch_idx, sample in enumerate(self.CustomDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)

                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()

                    loss = self.criterionL1(logits, labels)
                    T_loss = T_loss + loss.item()
                    break
            print(f"Loss Test: {T_loss/n_steps}")

        self.model.train()
        return T_loss/n_steps

    def SaveTestInference(self,save_path,custom=False):
        self.model.eval()
        inference = []
        with torch.no_grad():
            T_loss=0
            n_steps=0
            if custom==False:
                for batch_idx, sample in enumerate(self.TestDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)
                    inputs=inputs.unsqueeze(1)
                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()



                    loss = self.criterionL1(logits, labels)
                    inference.append(logits.to("cpu").numpy())
                    T_loss = T_loss + loss.item()
            else:
                for batch_idx, sample in enumerate(self.CustomDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)

                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()

                    loss = self.criterionL1(logits, labels)
                    T_loss = T_loss + loss.item()
                    break
            print(f"Loss Test: {T_loss/n_steps}")
        print(len(inference))
        inference = np.array(inference)
        print(inference.shape)
        np.save(save_path,inference)
        self.model.train()
        return T_loss/n_steps

    def TestInference(self,custom=False):
        self.model.eval()
        with torch.no_grad():
            T_loss=0
            n_steps=0
            if custom==False:
                for batch_idx, sample in enumerate(self.TestDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)
                    inputs=inputs.unsqueeze(1)
                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()



                    loss = self.criterionL1(logits, labels)
                    T_loss = T_loss + loss.item()
            else:
                for batch_idx, sample in enumerate(self.CustomDataset):
                    n_steps = n_steps + 1
                    inputs = sample["features"].to(self.device)

                    labels = sample["labels"].to(self.device)

                    outputs = self.model(inputs=inputs.to(self.device))
                    logits = outputs.logits.squeeze()

                    loss = self.criterionL1(logits, labels)
                    T_loss = T_loss + loss.item()
                    break
            print(f"Loss Test: {T_loss/n_steps}")

        self.model.train()
        return T_loss/n_steps

    def LoadDataset(self,path,name,trainable=True):

        if trainable == True:
            a = DatasetHandler.MeshDataset("train",path,"train_"+name+".npy")
            self.TrainDataset = DataLoader(a, batch_size=self.batch_sz, shuffle=True,drop_last=True)

            b = DatasetHandler.MeshDataset("test",path,"test_"+name+".npy")
            self.TestDataset = DataLoader(b, batch_size=self.batch_sz, shuffle=False,drop_last=False)
        else:
            a = DatasetHandler.MeshDataset("custom",name)
            self.CustomDataset = DataLoader(a, batch_size=2, shuffle=False,drop_last=True)

    def TestLoss(self,name=None):
        n_step=0
        total_loss=0

        if name==None:
            for batch_idx_j, sample_j in enumerate(self.CustomDataset):
                labels = sample_j["charges"].to(self.device)
                if len(labels) > 1:
                    n_step=n_step+1
                    print(sample_j["path"][0], sample_j["path"][1])
                    total_loss = total_loss + self.criterion(labels[0], labels[1]).item()
                    print(f"Loss short mapping: {self.criterion(labels[0], labels[1]).item()}")
                    print(f"Average loss short mapping: {total_loss/n_step}")
        if name=="Train":
            for batch_idx_j, sample_j in enumerate(self.TrainDataset):
                labels = sample_j["charges"].to(self.device)
                if len(labels) > 1:
                    n_step=n_step+1
                    total_loss = total_loss + self.criterion(labels[0], labels[1]).item()
                    print(f"Loss short mapping: {self.criterion(labels[0], labels[1]).item()}")
                    print(f"Average loss short mapping: {total_loss/n_step}")
        if name=="Test":
            for batch_idx_j, sample_j in enumerate(self.TestDataset):
                labels = sample_j["charges"].to(self.device)
                if len(labels) > 1:
                    n_step=n_step+1
                    total_loss = total_loss + self.criterion(labels[0], labels[1]).item()
                    print(f"Loss short mapping: {self.criterion(labels[0], labels[1]).item()}")
                    print(f"Average loss short mapping: {total_loss/n_step}")
        total_loss_short=total_loss/n_step
        n_step=0
        total_loss=0


        if name==None:
            for batch_idx_j, sample_j in enumerate(self.CustomDataset):
                n_step=n_step+1
                avg_charge=sample_j["charges_mask"].to(self.device)
                avg_charge=avg_charge*0
                labels = sample_j["charges"].to(self.device)

                total_loss = total_loss + self.criterion(avg_charge, labels).item()
                print(f"Loss 4 mapping: {self.criterion(avg_charge, labels).item()}")
                print(f"Average loss 4 mapping: {total_loss/n_step}")
        if name=="Train":
            for batch_idx_j, sample_j in enumerate(self.TrainDataset):
                n_step=n_step+1
                avg_charge=sample_j["charges_mask"].to(self.device)
                avg_charge=avg_charge*0
                labels = sample_j["charges"].to(self.device)

                total_loss = total_loss + self.criterion(avg_charge, labels).item()
                print(f"Loss 4 mapping: {self.criterion(avg_charge, labels).item()}")
                print(f"Average loss 4 mapping: {total_loss/n_step}")
        if name=="Test":
            for batch_idx_j, sample_j in enumerate(self.TestDataset):
                n_step=n_step+1
                avg_charge=sample_j["charges_mask"].to(self.device)
                avg_charge=avg_charge*0
                labels = sample_j["charges"].to(self.device)

                total_loss = total_loss + self.criterion(avg_charge, labels).item()
                print(f"Loss 4 mapping: {self.criterion(avg_charge, labels).item()}")
                print(f"Average loss 4 mapping: {total_loss/n_step}")

        total_loss_4=total_loss/n_step

        print(f"Average loss short mapping: {total_loss_short}, Average loss 4 mapping: {total_loss_4}")

    def Save(self,name):
        if name.startswith("SavedModels/"):
            name = name[12:]
        print("Saving model "+name)
        torch.save(self.model.state_dict(),"SavedModels/"+name+".pt")

    def Load(self,name):
        if name.startswith("SavedModels/"):
            name = name[12:]
            print("Loading model..")
            self.minloss=float(name[-14:])
            print(self.minloss)
            self.model.load_state_dict(torch.load("SavedModels/"+name+".pt"))
        else:
            self.model.load_state_dict(torch.load(name))

    def _setPreproccesor(self):

        self.preprocessor=PerceiverImagePreprocessor(
            self.config,
            prep_type="conv1x1",
            spatial_downsample=1,
            out_channels=256,
            position_encoding_type="trainable",
            concat_or_add_pos="concat",
            in_channels=1,
            project_pos_dim=256,
            trainable_position_encoding_kwargs=dict(num_channels=256, index_dims=self.config.d_model),
        )
        print("image size "+str(self.config.image_size))

    def _setDecoder(self):

        self.decoder = PerceiverBasicDecoder(
        self.config,
        num_channels=self.config.d_latents,
        # num_channels=200,-
        trainable_position_encoding_kwargs=dict(num_channels=self.config.d_latents, index_dims=self.config.d_model),
        use_query_residual=True,
        final_project=False,
        output_num_channels=1,
        num_heads=1,)

    def _setPostprocessor(self):

        self.postprocessor=PerceiverProjectionPostprocessor(
            in_channels=self.config.d_latents,
            out_channels=1,
            )



    def _setModel(self):
        if self.postprocessor != None and self.preprocessor == None:
            print("Using projection postprocessor..")
            self.model = PerceiverModel(self.config, decoder=self.decoder,output_postprocessor=self.postprocessor)
        if self.postprocessor == None  and self.preprocessor != None:
            print("Using image preprocessor..")
            self.model = PerceiverModel(self.config,input_preprocessor=self.preprocessor, decoder=self.decoder)
        if self.postprocessor == None and self.preprocessor == None:
            print("Using no preprocessors or postprocessors..")
            self.model = PerceiverModel(self.config, decoder=self.decoder)
        if self.postprocessor != None  and self.preprocessor != None:
            print("Using projection postprocessor and image preprocessor..")
            self.model = PerceiverModel(self.config, decoder=self.decoder,input_preprocessor=self.preprocessor,output_postprocessor=self.postprocessor)
        self.model.to(self.device)
        num_params = sum(param.numel() for param in self.model.parameters())
        print("num params "+ str(num_params))
