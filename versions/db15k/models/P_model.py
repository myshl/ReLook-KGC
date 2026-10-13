import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig
from helper import get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.prompter import Prompter
from models.bert_for_layerwise import BertModelForLayerwise
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans,DBSCAN


# import h5py
import numpy as np
import pickle
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.colors import Normalize


class KGCPromptTuner(pl.LightningModule):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters(ignore=['text_dict', 'gt'])
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)




        print(self.ent_embed.weight.shape)
        feature_dir = Path(__file__).resolve().parents[1] / 'img_data'
        with open(feature_dir / f'{self.configs.dataset}_img_features.pkl', 'rb') as file:
            img_features = pickle.load(file)
        img_features = F.normalize(torch.Tensor(img_features), p=2, dim=1)
        img_pool = torch.nn.AvgPool2d(4, stride=4)
        img = img_pool(img_features.view(-1, 64, 64))
        img = img.view(img.size(0), -1)

        print(img.shape)

        self.img_ent_embed = torch.nn.Embedding.from_pretrained(img, freeze=False)

        # print(self.img_ent_embed.weight.shape)


        # self.gat_ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)
        with open(feature_dir / f'{self.configs.dataset}_entity_embed.pkl', 'rb') as file:
            gat_entity = torch.as_tensor(pickle.load(file), dtype=torch.float32)
        with open(feature_dir / f'{self.configs.dataset}_relation_embed.pkl', 'rb') as file:
            gat_relation = torch.as_tensor(pickle.load(file), dtype=torch.float32)

        self.gat_ent_embed = nn.Embedding.from_pretrained(gat_entity, freeze=False)

        



        if self.configs.graph_model in ['transe', 'rotate']:
            self.img_rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
            self.gat_rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.img_rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)
            # self.gat_rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)
            self.gat_rel_embed = nn.Embedding.from_pretrained(gat_relation, freeze=False)





        if self.configs.graph_model in ['transe', 'rotate']:
            self.rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)

        self.img_prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.gat_prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)




        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)

        # print(self.fc.weight.shape)
        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)
        if configs.n_lar > 0:
            self.lar_loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=configs.gamma,
                distance_function=lambda x, y: self.graph_model.score_fn(x, y[0], y[1]),
            )

        self.history = {'perf': ..., 'loss': []}
        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        if self.configs.alpha_step > 0:
            self.alpha = 0.
        else:
            self.alpha = self.configs.alpha


        self.all_embeddings = []  # To store embeddings from all batches
        self.all_triples = []  # To store corresponding triples


    def forward(self, ent_rel, src_ids, src_mask):
        bs = ent_rel.size(0)
        all_ent_embed = self.gat_ent_embed.weight


        img_ent_embed = self.img_ent_embed.weight
        gat_ent_embed = self.gat_ent_embed.weight

        if self.configs.graph_model in ['transe', 'rotate']:
            all_rel_embed = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)
            img_rel_embed = torch.cat([self.img_rel_embed.weight, -self.img_rel_embed.weight], dim=0)
            gat_rel_embed = torch.cat([self.gat_rel_embed.weight, -self.gat_rel_embed.weight], dim=0)

        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            all_rel_embed = self.rel_embed.weight
            img_rel_embed = self.img_rel_embed.weight
            gat_rel_embed = self.gat_rel_embed.weight


        ent, rel = ent_rel[:, 0], ent_rel[:, 1]

        img_ent_embed = img_ent_embed[ent]
        img_rel_embed = img_rel_embed[rel]
        img_prompt = self.img_prompter(torch.stack([img_ent_embed, img_rel_embed], dim=1))

        gat_ent_embed = gat_ent_embed[ent]
        gat_rel_embed = gat_rel_embed[rel]
        gat_prompt = self.gat_prompter(torch.stack([gat_ent_embed, gat_rel_embed], dim=1))

        prompt = torch.cat([img_prompt, gat_prompt], dim=1)

        prompt_attention_mask = torch.ones(img_ent_embed.size(0), self.configs.prompt_length * 2).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)

        # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * 2]
        plm_ent_embed, plm_rel_embed = torch.chunk(ent_rel_state, chunks=2, dim=1)
        plm_ent_embed = self.fc(plm_ent_embed.reshape(img_ent_embed.size(0), -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(img_rel_embed.size(0), -1))

        # pred -- .shape: (batch_size, embed_dim)
        pred = self.graph_model(plm_ent_embed, plm_rel_embed)
        # logits -- .shape: (batch_size, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)
        
        return logits, img_ent_embed, plm_ent_embed, pred 

    def training_step(self, batched_data, batch_idx):
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        tgt_ent = batched_data['tgt_ent']
        labels = batched_data['labels']
        lars = batched_data['lars'] if self.configs.n_lar > 0 else None

        logits, _, _, pred = self(ent_rel, src_ids, src_mask)
        loss = self.loss_fn(logits, labels)
        if self.configs.n_lar > 0:
            lar_ent_embed = self.gat_ent_embed
            # pos, neg -- .shape: (batch_size, 1, embed_dim), pos_bias, neg_bias -- .shape: (batch_size, 1)
            pos, lar = lar_ent_embed(labels).unsqueeze(1), torch.mean(lar_ent_embed(lars), dim=1, keepdim=True)
            pos_bias, lar_bias = self.graph_model.bias[labels].unsqueeze(-1), torch.mean(self.graph_model.bias[lars], dim=-1, keepdim=True)
            lar_loss = self.lar_loss_fn(anchor=pred, positive=(pos, pos_bias), negative=(lar, lar_bias))
            loss = loss + self.alpha * lar_loss

        self.history['loss'].append(loss.detach().item())



        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # test_triples .shape: (batch_size, 3)
        test_triples = batched_data['triple']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # tgt_ent -- .type: list
        tgt_ent = batched_data['tgt_ent']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        # logits, prompt, ent_rel_state, pred 
        logits, img_ent_embed, plm_ent_embed, pred = self(ent_rel, src_ids, src_mask)

        # logits, pred = self(ent_rel, src_ids, src_mask)
        logits = logits.detach()


        # print(batch_idx)
            # --- Added functionality for embedding visualization ---
        if self.configs.visualize_embeddings:
            self.all_embeddings.append(pred.detach().cpu())
        if self.configs.visualize_embeddings and batch_idx % 70 == 0:
            # Extract the entity embeddings (using gat_ent_embed as an example)
            # entity_embeddings = logits.cpu().numpy()
            print(self.rel_embed.weight.shape)
            entity_embeddings = torch.cat(self.all_embeddings, dim=0).numpy()

            if len(entity_embeddings) <= 30:
                return self._filtered_ranks(logits, src_ent, rel, tgt_ent, test_triples, gt, batch_idx)


            # Dimensionality reduction (e.g., PCA or t-SNE)
            if self.configs.embedding_dim_reduction == 'PCA':
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(entity_embeddings)
            elif self.configs.embedding_dim_reduction == 'tSNE':
                tsne = TSNE(n_components=2, random_state=42)
                reduced_embeddings = tsne.fit_transform(entity_embeddings)
            else:
                raise ValueError("Invalid embedding_dim_reduction method. Choose 'PCA' or 'tSNE'.")
            # if batch_idx % 10 == 0:
            # # Store embeddings and their corresponding triples
            #     self.all_embeddings.append(reduced_embeddings)
            #     self.all_triples.append(test_triples)

            self.save_embedding_plot(reduced_embeddings)
            



        return self._filtered_ranks(logits, src_ent, rel, tgt_ent, test_triples, gt, batch_idx)

    def _filtered_ranks(self, logits, src_ent, rel, tgt_ent, test_triples, gt, batch_idx):
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt[(hi, ri)]
            ## store target score
            tgt_score = logits[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            ## recover the target values
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, argsort, ranks, batch_idx)
        return ranks

    def validation_epoch_end(self, outs):
        tail_ranks = np.concatenate(outs[0])
        head_ranks = np.concatenate(outs[1])

        perf = get_performance(self, tail_ranks, head_ranks)
        self.history['perf'] = perf
        print('Epoch:', self.current_epoch)
        print(perf)

    def on_validation_epoch_start(self):
        self.all_embeddings = []

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.configs.lr)

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx):
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                triple_str = self.ent_names[head] + ' [' + self.ent_descs[head] + '] | ' + self.rel_names[rel]\
                    + ' | ' + self.ent_names[tail] + ' [' + self.ent_descs[tail] + '] ' + timestamp + '(%d %d %d)' % (head, tail, rel)
                file.write(str(batch_idx * self.configs.val_batch_size + i) + '. ' + triple_str + '=> ranks: ' + str(rank) + '\n')

                best10 = argsort[i, :10]
                for ii, ent in enumerate(best10):
                    ent = ent.item()
                    mark = '*' if (ii + 1) == rank else ' '
                    file.write('\t%2d%s ' % (ii + 1, mark) + self.ent_names[ent] + ' [' + self.ent_descs[ent] + ']' + ' (%d)' % ent + '\n')

    # def save_embedding_plot(self, reduced_embeddings, test_triples):
    #     """
    #     Saves a 2D plot of entity embeddings after dimensionality reduction (PCA/tSNE) for all batches.
    #     The plot colors the points based on their density.
    #     """
    #     plt.figure(figsize=(10, 10))

    #     # Calculate point density using KernelDensity (you can also use other density estimators)
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.2)  # You may want to experiment with the bandwidth
    #     kde.fit(reduced_embeddings)
    #     densities = np.exp(kde.score_samples(reduced_embeddings))  # Compute density at each point

    #     # Plot the reduced embeddings with color based on density
    #     scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=densities, cmap='viridis', s=10, alpha=0.7)

    #     # Add a colorbar to indicate density
    #     plt.colorbar(scatter, label="Density")

    #     # # Optionally add labels to the points (e.g., for debugging purposes)
    #     # for i, triple in enumerate(test_triples):
    #     #     # Annotate some points to help identify them
    #     #     if i % 50 == 0:  # Annotate every 50th point for clarity
    #     #         plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], f"{triple[0]}-{triple[1]}", fontsize=8)

    #     # Add title and labels
    #     plt.title("Entity Embeddings (All Batches) with Density Coloring")
    #     plt.xlabel("PCA/tSNE Component 1")
    #     plt.ylabel("PCA/tSNE Component 2")

    #     # Set fixed axis limits
    #     plt.xlim(-10, 10)
    #     plt.ylim(-10, 10)

    #     # Save the plot
    #     save_path = os.path.join(self.configs.img_save_dir, f"all_entity_embeddings_with_density-111.png")
    #     plt.savefig(save_path)
    #     plt.close()

    def save_embedding_plot(self, reduced_embeddings):
        """
        Saves a 2D plot of entity embeddings after dimensionality reduction (PCA/tSNE) for all batches.
        The plot colors the points based on their density, with uniform scaling.
        """
        plt.figure(figsize=(15, 15))

        # Calculate point density using KernelDensity (you can also use other density estimators)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)  # You may want to experiment with the bandwidth
        kde.fit(reduced_embeddings)
        densities = np.exp(kde.score_samples(reduced_embeddings))  # Compute density at each point

        # Normalize densities to [0, 1] range
        norm = Normalize(vmin=np.min(densities), vmax=np.max(densities))
        normalized_densities = norm(densities)

        # Plot the reduced embeddings with color based on normalized density
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=normalized_densities, cmap='viridis', s=10, alpha=0.7)

        # Add a colorbar to indicate density
        plt.colorbar(scatter, label="Density (Normalized)")


        # Add title and labels
        # plt.title("Entity Embeddings (All Batches) with Normalized Density Coloring")
        # plt.xlabel("PCA/tSNE Component 1")
        # plt.ylabel("PCA/tSNE Component 2")

        # Set fixed axis limits
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)

        # Save the plot
        save_path = os.path.join(self.configs.img_save_dir, f"all_entity_embeddings_with_normalized_density.svg")
        plt.savefig(save_path)
        plt.close()

    # def save_embedding_plot(self, reduced_embeddings, eps=0.5, min_samples=5):
    #     """
    #     Saves a 2D plot of entity embeddings after dimensionality reduction (PCA/tSNE) for all batches.
    #     The plot colors the points based on DBSCAN clustering, keeping only the dense regions (clusters).
        
    #     :param reduced_embeddings: The 2D array of reduced entity embeddings.
    #     :param eps: The maximum distance between two points to be considered as in the same neighborhood (DBSCAN parameter).
    #     :param min_samples: The number of points required to form a dense region (DBSCAN parameter).
    #     """
    #     plt.figure(figsize=(15, 15))

    #     # Apply DBSCAN clustering to the reduced embeddings
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #     cluster_labels = dbscan.fit_predict(reduced_embeddings)  # Get cluster labels for each point

    #     # Only keep points that are part of a cluster (not labeled as -1)
    #     mask = cluster_labels != -1
    #     clustered_embeddings = reduced_embeddings[mask]
    #     clustered_labels = cluster_labels[mask]

    #     # Plot the clustered embeddings with colors corresponding to DBSCAN cluster labels
    #     scatter = plt.scatter(clustered_embeddings[:, 0], clustered_embeddings[:, 1], c=clustered_labels, cmap='viridis', s=10, alpha=0.7)

    #     # Add a colorbar to indicate cluster membership
    #     plt.colorbar(scatter, label="Cluster ID")

    #     # Add title and labels
    #     plt.title(f"Entity Embeddings (All Batches) with DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    #     plt.xlabel("PCA/tSNE Component 1")
    #     plt.ylabel("PCA/tSNE Component 2")

    #     # Set fixed axis limits (optional)
    #     # plt.xlim(-5, 5)
    #     # plt.ylim(-5, 5)

    #     # Save the plot
    #     save_path = os.path.join(self.configs.img_save_dir, f"all_entity_embeddings_with_dbscan_clusters_eps_{eps}_min_samples_{min_samples}.svg")
    #     plt.savefig(save_path)
    #     plt.close()




    # def save_embedding_plot(self, reduced_embeddings, density_threshold=0.01):
    #     """
    #     Saves a 2D plot of entity embeddings after dimensionality reduction (PCA/tSNE) for all batches.
    #     The plot colors the points based on their density, with uniform scaling.
    #     Points with density below the given threshold will be excluded.
    #     """
    #     plt.figure(figsize=(15, 15))

    #     # Calculate point density using KernelDensity (you can also use other density estimators)
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.2)  # You may want to experiment with the bandwidth
    #     kde.fit(reduced_embeddings)
    #     densities = np.exp(kde.score_samples(reduced_embeddings))  # Compute density at each point

    #     print(densities)

    #     # Filter out points with density below the threshold
    #     mask = densities > density_threshold
    #     filtered_embeddings = reduced_embeddings[mask]
    #     filtered_densities = densities[mask]

    #     if len(filtered_densities) == 0:
    #         print("Warning: No points above the density threshold. Skipping plot.")
    #         return  # Skip plotting if no points remain

    #     # Normalize densities to [0, 1] range
    #     norm = Normalize(vmin=np.min(filtered_densities), vmax=np.max(filtered_densities))
    #     normalized_densities = norm(filtered_densities)

    #     # Plot the reduced embeddings with color based on normalized density
    #     scatter = plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1], 
    #                         c=normalized_densities, cmap='viridis', s=10, alpha=0.7)

    #     # Add a colorbar to indicate density
    #     plt.colorbar(scatter, label="Density (Normalized)")

    #     # Add title and labels
    #     plt.title("Entity Embeddings (All Batches) with Normalized Density Coloring")
    #     plt.xlabel("PCA/tSNE Component 1")
    #     plt.ylabel("PCA/tSNE Component 2")

    #     # Set fixed axis limits (optional)
    #     # plt.xlim(-5, 5)
    #     # plt.ylim(-5, 5)

    #     # Save the plot as a vector graphic (SVG or PDF)
    #     save_path = os.path.join(self.configs.img_save_dir, f"all_entity_embeddings_with_normalized_density.svg")
    #     plt.savefig(save_path, format='svg')
    #     plt.close()

    # def save_embedding_plot(self, reduced_embeddings, density_threshold=0.1, distance_threshold=0.5):
    #     """
    #     Saves a 2D plot of entity embeddings after dimensionality reduction (PCA/tSNE) for all batches.
    #     The plot colors the points based on their density, with uniform scaling.
    #     Points with density below the given threshold will be excluded.
    #     Additionally, points surrounding high-density points will also be plotted to show clusters.
    #     """
    #     plt.figure(figsize=(15, 15))

    #     # Calculate point density using KernelDensity (you can also use other density estimators)
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.2)  # You may want to experiment with the bandwidth
    #     kde.fit(reduced_embeddings)
    #     densities = np.exp(kde.score_samples(reduced_embeddings))  # Compute density at each point

    #     # Filter out points with density below the threshold
    #     mask = densities > density_threshold
    #     filtered_embeddings = reduced_embeddings[mask]
    #     filtered_densities = densities[mask]

    #     # Check if filtered_densities is empty
    #     if len(filtered_densities) == 0:
    #         print("Warning: No points above the density threshold. Skipping plot.")
    #         return  # Skip plotting if no points remain

    #     # Normalize densities to [0, 1] range
    #     norm = Normalize(vmin=np.min(filtered_densities), vmax=np.max(filtered_densities))
    #     normalized_densities = norm(filtered_densities)

    #     # Plot the high-density points
    #     scatter = plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1], 
    #                         c=normalized_densities, cmap='viridis', s=10, alpha=0.7)

    #     # Add a colorbar to indicate density
    #     plt.colorbar(scatter, label="Density (Normalized)")

    #     # For each high-density point, find its neighbors within a distance threshold
    #     distances = euclidean_distances(filtered_embeddings)
    #     print(distances.shape)
    #     print(distances)

    #     for i, point in enumerate(filtered_embeddings):
    #         # Find all points within a certain distance from the high-density point
    #         nearby_points = np.where(distances[i] < distance_threshold)[0]
    #         plt.scatter(filtered_embeddings[nearby_points, 0], filtered_embeddings[nearby_points, 1], 
    #                     c='gray', s=10, alpha=0.3)

    #     # Add title and labels
    #     plt.title("Entity Embeddings (All Batches) with Normalized Density Coloring and Cluster Visualization")
    #     plt.xlabel("PCA/tSNE Component 1")
    #     plt.ylabel("PCA/tSNE Component 2")

    #     # Set fixed axis limits (optional)
    #     # plt.xlim(-5, 5)
    #     # plt.ylim(-5, 5)

    #     # Save the plot as a vector graphic (SVG or PDF)
    #     save_path = os.path.join(self.configs.img_save_dir, f"all_entity_embeddings_with_clusters_and_density.svg")
    #     plt.savefig(save_path, format='svg')
    #     plt.close()
