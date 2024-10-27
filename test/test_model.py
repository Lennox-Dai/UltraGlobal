# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import torch
import numpy as np

import time

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA
import torch 
import pdb

def save_features(features, file_path):
    torch.save(features, file_path)


def load_features(file_path):
    return torch.load(file_path)


def save_features(features, file_path):
    print(f"Saving features to {file_path}")
    torch.save(features, file_path)


def load_features(file_path):
    print(f"Loading features from {file_path}")
    return torch.load(file_path)

@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list, is_rerank, gemp, rgem, sgem, onemeval, depth, logger):
    torch.backends.cudnn.benchmark = False
    model.eval()
    
    # pdb.set_trace()
    state_dict = model.state_dict()

    # initialize modules
    MDescAug_obj = MDescAug()
    RerankwMDA_obj = RerankwMDA()

    model.load_state_dict(state_dict)
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset

        # # 构建唯一的文件路径，包含数据集名称和模型深度等信息
        # query_features_path = f"{dataset}_query_features.pth"
        # db_features_path = f"{dataset}_db_features.pth"

        # if os.path.exists(query_features_path) and os.path.exists(db_features_path):
        #     print("Loading saved features")
        #     Q = load_features(query_features_path)
        #     X = load_features(db_features_path)
        # else:
        #     print("Extracting query features")
        #     Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
        #     print("Extracting database features")
        #     X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)
        #     save_features(Q, query_features_path)
        #     save_features(X, db_features_path)

        print("extract query features")
        Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
        print("extract database features")
        X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)

        cfg = config_gnd(dataset,data_dir)
        Q = torch.tensor(Q).cuda()
        X = torch.tensor(X).cuda()
        
        print("perform global feature reranking")
        if onemeval:
            X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
            X = torch.concat([X,X_expand],0)
        sim = torch.matmul(X, Q.T) # 6322 70
        ranks = torch.argsort(-sim, axis=0) # 6322 70
        if is_rerank:
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()
        # pdb.set_trace()
        
        # revisited evaluation
        ks = [1, 5, 10]
        # (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])
        (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('Retrieval results: aps E: {}, M: {}, H: {}'.format(np.around(apsE*100, decimals=2), np.around(apsM*100, decimals=2), np.around(apsH*100, decimals=2)))
        print('Retrieval results: mpr E: {}, M: {}, H: {}'.format(np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
        print('Retrieval results: prs E: {}, M: {}, H: {}'.format(np.around(prsE*100, decimals=2), np.around(prsM*100, decimals=2), np.around(prsH*100, decimals=2)))
        logger.info('===============================================================================================')
        logger.info('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        logger.info('===============================================================================================')
        logger.info('Retrieval results: aps E: {}, M: {}, H: {}'.format(np.around(apsE*100, decimals=2), np.around(apsM*100, decimals=2), np.around(apsH*100, decimals=2)))
        logger.info('===============================================================================================')
        logger.info('Retrieval results: mpr E: {}, M: {}, H: {}'.format(np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
        logger.info('===============================================================================================')
        logger.info('Retrieval results: prs E: {}, M: {}, H: {}'.format(np.around(prsE*100, decimals=2), np.around(prsM*100, decimals=2), np.around(prsH*100, decimals=2)))
        # logger.info('===============================================================================================')
        
# written by Seongwon Lee (won4113@yonsei.ac.kr)

# import os
# import torch
# import numpy as np
# import cv2
# import os
# from sklearn.decomposition import PCA as sklearnPCA


# import time

# from test.config_gnd import config_gnd
# from test.test_utils import extract_feature, test_revisitop

# from modules.reranking.MDescAug import MDescAug
# from modules.reranking.RerankwMDA import RerankwMDA
# import torch 
# import pdb

# COLUMNOFCODEBOOK = 32
# DESDIM = 128
# SUBVEC = 32
# SUBCLUSTER = 256
# PCAD = 128
# TESTTYPE = 0

# def save_features(features, file_path):
#     torch.save(features, file_path)


# def load_features(file_path):
#     return torch.load(file_path)


# def save_features(features, file_path):
#     print(f"Saving features to {file_path}")
#     torch.save(features, file_path)


# def load_features(file_path):
#     print(f"Loading features from {file_path}")
#     return torch.load(file_path)

# @torch.no_grad()
# def test_model(model, data_dir, dataset_list, scale_list, is_rerank, gemp, rgem, sgem, onemeval, depth, logger):
#     torch.backends.cudnn.benchmark = False
#     model.eval()
    
#     state_dict = model.state_dict()
    

#     # initialize modules
#     MDescAug_obj = MDescAug()
#     RerankwMDA_obj = RerankwMDA()



#     model.load_state_dict(state_dict)
#     for dataset in dataset_list:
#         text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
#         print(text)
#         if dataset == 'roxford5k':
#             gnd_fn = 'gnd_roxford5k.pkl'
#         elif dataset == 'rparis6k':
#             gnd_fn = 'gnd_rparis6k.pkl'
#         else:
#             assert dataset

#         # 构建唯一的文件路径，包含数据集名称和模型深度等信息
#         query_features_path = f"{dataset}_query_features.pth"
#         db_features_path = f"{dataset}_db_features.pth"

#         if os.path.exists(query_features_path) and os.path.exists(db_features_path):
#             print("Loading saved features")
#             Q = load_features(query_features_path)
#             X = load_features(db_features_path)
#         else:
#             print("Extracting query features")
#             Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
#             print("Extracting database features")
#             X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)
#             save_features(Q, query_features_path)
#             save_features(X, db_features_path)


#         # print("extract query features")
#         # Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
#         # print("extract database features")
#         # X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)

#         cfg = config_gnd(dataset,data_dir)
#         Q = torch.tensor(Q).cuda()
#         X = torch.tensor(X).cuda()

#         ####
#         Q = encode_single_image(Q)
#         X = encode_single_image(X)
#         ####


#         print("perform global feature reranking")
#         if onemeval:
#             X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
#             X = torch.concat([X,X_expand],0)
#         sim = torch.matmul(X, Q.T) # 6322 70
#         ranks = torch.argsort(-sim, axis=0) # 6322 70
#         if is_rerank:
#             rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
#             ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
#         ranks = ranks.data.cpu().numpy()
#         pdb.set_trace()

#         # revisited evaluation
#         ks = [1, 5, 10]
#         (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

#         print('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
#         logger.info('Retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        


# def encode_single_image(features_tensor, codebook_clusters=32, use_pca=False, pca_components=128):
#     """
#     Description: Encode a single image's features using VLAD with optional PCA.
#     Input: 
#         features_tensor - A tensor of the image's descriptors.
#         codebook_clusters - Number of clusters for the codebook (default: 32).
#         use_pca - Boolean to indicate whether to use PCA for dimensionality reduction (default: False).
#         pca_components - Number of PCA components if use_pca is True (default: 128).
#     Output: 
#         encoded_vector - Encoded VLAD vector of the image.
#         pca_model - Trained PCA model if use_pca is True, otherwise None.
#     """
#     # Convert tensor to numpy array
#     features_array = features_tensor.numpy()
    
#     # Step 1: Train the codebook
#     NNlabel, codebook = get_codebook(features_array, codebook_clusters)
    
#     if use_pca:
#         # Step 2: Compute VLAD vectors with PCA
#         encoded_vector, sk_pca = get_vlad_single_image_pca(features_array, NNlabel, codebook, pca_components)
#         return torch.tensor(encoded_vector).cuda(), sk_pca
#     else:
#         # Step 2: Compute VLAD vectors without PCA
#         encoded_vector = get_vlad_single_image(features_array, NNlabel, codebook)
#         return torch.tensor(encoded_vector).cuda(), None

# def get_vlad_single_image(features, NNlabel, codebook):
#     """
#     Description: Compute VLAD vector for a single image without PCA.
#     Input: 
#         features - Descriptors of the image.
#         NNlabel - Codebook cluster assignments for descriptors.
#         codebook - Codebook centroids.
#     Output: 
#         vlad_base - VLAD vector of the image.
#     """
#     vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
#     for eachDes in range(features.shape[0]):
#         des = features[eachDes]
#         centriods_id = NNlabel[eachDes]
#         centriods = codebook[centriods_id]
    
#         vlad[centriods_id] = vlad[centriods_id] + des - centriods
    
#     vlad_norm = vlad.copy()
#     cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
#     return vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)

# def get_vlad_single_image_pca(features, NNlabel, codebook, pca_components):
#     """
#     Description: Compute VLAD vector for a single image with PCA.
#     Input: 
#         features - Descriptors of the image.
#         NNlabel - Codebook cluster assignments for descriptors.
#         codebook - Codebook centroids.
#         pca_components - Number of PCA components.
#     Output: 
#         vlad_base_pca - VLAD vector of the image after PCA.
#         sklearn_pca - Trained PCA model.
#     """
#     vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
#     for eachDes in range(features.shape[0]):
#         des = features[eachDes]
#         centriods_id = NNlabel[eachDes]
#         centriods = codebook[centriods_id]
    
#         vlad[centriods_id] = vlad[centriods_id] + des - centriods
    
#     vlad = vlad.reshape(-1, COLUMNOFCODEBOOK * DESDIM)

#      # Check the number of samples
#     n_samples = vlad.shape[0]
#     n_features = vlad.shape[1]
    
#     # Ensure n_components does not exceed available samples or features
#     if pca_components > min(n_samples, n_features):
#         pca_components = min(n_samples, n_features)


#     sklearn_pca = sklearnPCA(n_components=pca_components)
#     vlad_pca = sklearn_pca.fit_transform(vlad)
#     vlad_pca_norm = vlad_pca.copy()
#     cv2.normalize(vlad_pca, vlad_pca_norm, 1.0, 0.0, cv2.NORM_L2)
    
#     return vlad_pca_norm, sklearn_pca


# def get_codebook(all_des, K):
#     '''
#     Description: train the codebook from all of the descriptors
#     Input: all_des - training data for the codebook
#                  K - the column of the codebook

#     '''
#     label, center = get_cluster_center(all_des, K)
#     return label, center


# def get_cluster_center(des_set, K):
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
#     des_set = np.float32(des_set)
#     ret, label, center = cv2.kmeans(des_set, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
#     return label, center

# def cal_vec_dist(vec1, vec2):
#     return np.linalg.norm(vec1 - vec2)

# def get_pic_vlad(pic, des_size, codebook):
#     vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
#     for eachDes in range(des_size):
#         des = pic[eachDes]
#         min_dist = 1000000000.0
#         ind = 0
#         for i in range(COLUMNOFCODEBOOK):
#             dist = cal_vec_dist(des, codebook[i])
#             if dist < min_dist:
#                 min_dist = dist
#                 ind = i
#         vlad[ind] = vlad[ind] + des - codebook[ind]
    
#     vlad_norm = vlad.copy()
#     cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
#     vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)
    
#     return vlad_norm

# def get_pic_vlad_pca(pic, des_size, codebook, sklearn_pca):
#     vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
#     for eachDes in range(des_size):
#         des = pic[eachDes]
#         min_dist = 1000000000.0
#         ind = 0
#         for i in range(COLUMNOFCODEBOOK):
#             dist = cal_vec_dist(des, codebook[i])
#             if dist < min_dist:
#                 min_dist = dist
#                 ind = i
#         vlad[ind] = vlad[ind] + des - codebook[ind]
    
#     vlad = vlad.reshape(-1, COLUMNOFCODEBOOK * DESDIM)
#     sklearn_transf = sklearn_pca.transform(vlad)
#     sklearn_transf_norm = sklearn_transf.copy()
#     cv2.normalize(sklearn_transf, sklearn_transf_norm, 1.0, 0.0, cv2.NORM_L2)
    
#     return sklearn_transf_norm