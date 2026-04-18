'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import os
import numpy as np
import torch
import pandas as pd

cores = int(os.environ.get("CDD_TEST_CORES", "1"))

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def _load_case_study_user_ids():
    """Load user ids from rebuttal/mrr_case_study.csv (if present)."""
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        case_csv = os.path.join(repo_root, "rebuttal", "mrr_case_study.csv")
        if not os.path.exists(case_csv):
            return set()
        df = pd.read_csv(case_csv)
        if "user_id" not in df.columns:
            return set()
        return set(pd.to_numeric(df["user_id"], errors="coerce").dropna().astype(int).tolist())
    except Exception:
        return set()


CASE_STUDY_USER_IDS = _load_case_study_user_ids()

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc, K_max_item_score

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc, K_max_item_score

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio, mrr = [], [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))
        mrr.append(metrics.mrr_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio),
            'mrr': np.array(mrr), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc, top_items = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc, top_items = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    # Calculate hits (items in top K that are in test set)
    K_max = max(Ks)
    hits = [item for item in top_items[:K_max] if item in user_pos_test]
    
    perf = get_performance(user_pos_test, r, auc, Ks)
    # Add detailed information
    perf['user_id'] = u
    perf['training_items'] = training_items
    perf['test_items'] = user_pos_test
    perf['hits'] = hits
    perf['top_predicted'] = top_items[:K_max] if len(top_items) >= K_max else top_items

    # Lightweight: only compute full-list rank positions for selected case-study users.
    if int(u) in CASE_STUDY_USER_IDS:
        scores = np.asarray(rating)
        sorted_idx = np.argsort(-scores)  # descending over all ITEM_NUM items (e.g., 2000)
        rank_pos = np.empty_like(sorted_idx)
        rank_pos[sorted_idx] = np.arange(1, len(sorted_idx) + 1)
        perf['test_item_rank_in_2000'] = {
            int(item): int(rank_pos[int(item)])
            for item in user_pos_test
            if 0 <= int(item) < len(rank_pos)
        }
    
    return perf


def test(model, users_to_test, drop_flag=False, batch_test_flag=False, return_detailed=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'mrr': np.zeros(len(Ks)), 'auc': 0.}
    
    detailed_results = []  # Store individual patient results

    pool = multiprocessing.Pool(cores) if cores > 1 else None

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch.numpy()
                i_count += i_rate_batch.shape[1]
                
                # Clear GPU cache after each item batch to prevent memory overflow
                del u_g_embeddings, pos_i_g_embeddings, i_rate_batch
                torch.cuda.empty_cache()

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            
            # Clear GPU cache after processing user batch
            del u_g_embeddings, pos_i_g_embeddings
            torch.cuda.empty_cache()

        # Convert to numpy if rate_batch is a torch tensor
        if isinstance(rate_batch, torch.Tensor):
            rate_batch_np = rate_batch.numpy()
        else:
            rate_batch_np = rate_batch
        user_batch_rating_uid = zip(rate_batch_np, user_batch)
        if pool is not None:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
        else:
            batch_result = list(map(test_one_user, user_batch_rating_uid))
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['mrr'] += re['mrr']/n_test_users
            result['auc'] += re['auc']/n_test_users
            
            if return_detailed:
                detailed_results.append(re)


    assert count == n_test_users
    if pool is not None:
        pool.close()
    
    if return_detailed:
        return result, detailed_results
    return result


def generate_detailed_report(detailed_results, output_file='disease_prediction_result3.txt', csv_file='high_accuracy_patients.csv'):
    """
    Generate a detailed report file similar to the example format.
    
    Args:
        detailed_results: List of detailed patient results from test function
        output_file: Path to output text file
        csv_file: Path to output CSV file
    """
    # Find index for K=20, or use the last index if 20 not found
    target_K = 20
    if len(detailed_results) > 0 and len(detailed_results[0]['precision']) > 0:
        # Try to find K=20, default to last index
        k_idx = len(detailed_results[0]['precision']) - 1
        # If Ks is available, find the correct index
        if target_K in Ks:
            k_idx = Ks.index(target_K)
    else:
        k_idx = 0
    
    # Filter valid patients (those with test diseases)
    valid_patients = [r for r in detailed_results if len(r['test_items']) > 0]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Analysis complete! Found {len(valid_patients)} valid patients.\n\n")
        f.write("Top 10 patients by different metrics:\n\n")
        
        # Sort patients by different metrics
        sorted_by_precision = sorted(valid_patients, 
                                     key=lambda x: x['precision'][k_idx], 
                                     reverse=True)[:10]
        # Filter patients with test disease count >= 5, then sort by top 5 hit ratio and recall
        patients_with_5plus_test = [r for r in valid_patients if len(r['test_items']) >= 5]
        # Calculate top 5 predicted diseases hit ratio for each patient
        for r in patients_with_5plus_test:
            top5_pred = r['top_predicted'][:5] if len(r['top_predicted']) >= 5 else r['top_predicted']
            top5_hits = [item for item in top5_pred if item in r['test_items']]
            r['top5_hit_ratio'] = len(top5_hits) / 5.0 if len(top5_pred) > 0 else 0.0
        # Sort by top 5 hit ratio first, then by recall
        sorted_by_recall = sorted(patients_with_5plus_test, 
                                  key=lambda x: (x['top5_hit_ratio'], x['recall'][k_idx]), 
                                  reverse=True)
        sorted_by_ndcg = sorted(valid_patients, 
                                key=lambda x: x['ndcg'][k_idx], 
                                reverse=True)[:10]
        sorted_by_hit = sorted(valid_patients, 
                               key=lambda x: (x['hit_ratio'][k_idx], x['precision'][k_idx]), 
                               reverse=True)[:10]
        
        # Top 10 by Precision@20
        # f.write("=" * 60 + "\n")
        # f.write("TOP 10 PATIENTS BY PRECISION@20\n")
        # f.write("=" * 60 + "\n")
        # f.write(f"{'user_id':>8} {'precision':>10} {'recall':>10} {'ndcg':>10} {'num_hits':>10} {'num_test_diseases':>18}\n")
        # for r in sorted_by_precision:
        #     num_hits = len(r['hits'])
        #     num_test = len(r['test_items'])
        #     recall_str = f"{r['recall'][k_idx]:.1f}" if abs(r['recall'][k_idx] - 1.0) < 1e-6 else f"{r['recall'][k_idx]:.6f}"
        #     f.write(f"{r['user_id']:>8} {r['precision'][k_idx]:>10.2f} {recall_str:>10} {r['ndcg'][k_idx]:>10.6f} "
        #            f"{num_hits:>10} {num_test:>18}\n")
        
        # f.write("\n")
        
        # Top 10 by Recall@20 (with test disease count >= 5, sorted by top 5 hit ratio)
        f.write("=" * 60 + "\n")
        f.write("TOP 10 PATIENTS BY RECALL@20 (Test Disease Count >= 5, Sorted by Top 5 Hit Ratio)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'user_id':>8} {'precision':>10} {'recall':>10} {'ndcg':>10} {'top5_ratio':>12} {'num_hits':>10} {'num_test_diseases':>18}\n")
        for r in sorted_by_recall[:10]:
            num_hits = len(r['hits'])
            num_test = len(r['test_items'])
            recall_str = f"{r['recall'][k_idx]:.1f}" if abs(r['recall'][k_idx] - 1.0) < 1e-6 else f"{r['recall'][k_idx]:.6f}"
            top5_ratio = r.get('top5_hit_ratio', 0.0)
            f.write(f"{r['user_id']:>8} {r['precision'][k_idx]:>10.2f} {recall_str:>10} {r['ndcg'][k_idx]:>10.6f} "
                   f"{top5_ratio:>12.2f} {num_hits:>10} {num_test:>18}\n")
        
        # f.write("\n")
        
        # # Top 10 by NDCG@20
        # f.write("=" * 60 + "\n")
        # f.write("TOP 10 PATIENTS BY NDCG@20\n")
        # f.write("=" * 60 + "\n")
        # f.write(f"{'user_id':>8} {'precision':>10} {'recall':>10} {'ndcg':>10} {'num_hits':>10} {'num_test_diseases':>18}\n")
        # for r in sorted_by_ndcg:
        #     num_hits = len(r['hits'])
        #     num_test = len(r['test_items'])
        #     recall_str = f"{r['recall'][k_idx]:.1f}" if abs(r['recall'][k_idx] - 1.0) < 1e-6 else f"{r['recall'][k_idx]:.6f}"
        #     f.write(f"{r['user_id']:>8} {r['precision'][k_idx]:>10.2f} {recall_str:>10} {r['ndcg'][k_idx]:>10.6f} "
        #            f"{num_hits:>10} {num_test:>18}\n")
        
        f.write("\n")
        
        # # Top 10 by Hit Ratio@20
        # f.write("=" * 60 + "\n")
        # f.write("TOP 10 PATIENTS BY HIT RATIO@20\n")
        # f.write("=" * 60 + "\n")
        # f.write(f"{'user_id':>8} {'precision':>10} {'recall':>10} {'ndcg':>10} {'num_hits':>10} {'num_test_diseases':>18}\n")
        # for r in sorted_by_hit:
        #     num_hits = len(r['hits'])
        #     num_test = len(r['test_items'])
        #     f.write(f"{r['user_id']:>8} {r['precision'][k_idx]:>10.2f} {r['recall'][k_idx]:>10.6f} "
        #            f"{r['ndcg'][k_idx]:>10.6f} {num_hits:>10} {num_test:>18}\n")
        
        # f.write("\n")
        
        # Overall statistics
        avg_precision = np.mean([r['precision'][k_idx] for r in valid_patients])
        avg_recall = np.mean([r['recall'][k_idx] for r in valid_patients])
        avg_ndcg = np.mean([r['ndcg'][k_idx] for r in valid_patients])
        avg_hit = np.mean([r['hit_ratio'][k_idx] for r in valid_patients])
        patients_with_hits = sum(1 for r in valid_patients if r['hit_ratio'][k_idx] > 0)
        hit_percentage = (patients_with_hits / len(valid_patients)) * 100 if valid_patients else 0
        
        f.write("=" * 60 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Average Precision@20: {avg_precision:.4f}\n")
        f.write(f"Average Recall@20: {avg_recall:.4f}\n")
        f.write(f"Average NDCG@20: {avg_ndcg:.4f}\n")
        f.write(f"Average Hit Ratio@20: {avg_hit:.4f}\n")
        f.write(f"Patients with at least 1 hit: {patients_with_hits} ({hit_percentage:.1f}%)\n\n")

        # Case-study section: ranks of test diseases in the full 2000-d prediction list.
        case_results = [r for r in valid_patients if int(r['user_id']) in CASE_STUDY_USER_IDS]
        if case_results:
            f.write("=" * 80 + "\n")
            f.write("CASE STUDY: TEST DISEASE RANK IN FULL PREDICTION LIST (2000 ITEMS)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Users from rebuttal/mrr_case_study.csv: {len(case_results)}\n\n")
            for r in sorted(case_results, key=lambda x: int(x['user_id'])):
                f.write(f"Patient {int(r['user_id'])}:\n")
                rank_map = r.get('test_item_rank_in_2000', {})
                if rank_map:
                    for disease_id, rank in sorted(rank_map.items(), key=lambda kv: kv[1]):
                        f.write(f"  test_disease={disease_id}, rank_in_2000={rank}\n")
                else:
                    f.write("  rank data unavailable (user not processed for case-study ranks).\n")
                f.write("\n")
        
        # Detailed results for top 5 patients (by precision)
        # f.write("=" * 80 + "\n")
        # f.write("DETAILED RESULTS FOR TOP 5 PATIENTS\n")
        # f.write("=" * 80 + "\n\n")
        
        # top5 = sorted_by_precision[:5]
        # for idx, r in enumerate(top5, 1):
        #     f.write(f"{idx}. PATIENT {r['user_id']} (Precision: {r['precision'][k_idx]:.4f})\n")
        #     f.write("-" * 50 + "\n")
        #     f.write(f"Training diseases: {len(r['training_items'])}\n")
        #     f.write(f"Test diseases: {len(r['test_items'])}\n")
        #     f.write(f"Precision@20: {r['precision'][k_idx]:.4f}\n")
        #     f.write(f"Recall@20: {r['recall'][k_idx]:.4f}\n")
        #     f.write(f"NDCG@20: {r['ndcg'][k_idx]:.4f}\n")
        #     f.write(f"Hits: {r['hits']}\n")
        #     f.write(f"Test diseases: {sorted(r['test_items'])}\n")
        #     top5_pred = r['top_predicted'][:5] if len(r['top_predicted']) >= 5 else r['top_predicted']
        #     f.write(f"Top 5 predicted: {top5_pred}\n\n")
        
        # f.write(f"Results saved to {csv_file}\n\n")
        
        # Detailed results for top 5 patients by recall
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS FOR TOP 20 PATIENTS BY RECALL@20\n")
        f.write("=" * 80 + "\n\n")
        
        top20_recall = sorted_by_recall[:20]
        for idx, r in enumerate(top20_recall, 1):
            f.write(f"{idx}. PATIENT {r['user_id']} (Recall: {r['recall'][k_idx]:.4f})\n")
            f.write("-" * 50 + "\n")
            f.write(f"Training diseases: {len(r['training_items'])}\n")
            f.write(f"Test diseases: {len(r['test_items'])}\n")
            f.write(f"Precision@20: {r['precision'][k_idx]:.4f}\n")
            f.write(f"Recall@20: {r['recall'][k_idx]:.4f}\n")
            f.write(f"NDCG@20: {r['ndcg'][k_idx]:.4f}\n")
            f.write(f"Hit Ratio@20: {r['hit_ratio'][k_idx]:.4f}\n")
            f.write(f"Hits (correctly predicted diseases): {sorted(r['hits'])}\n")
            f.write(f"All test diseases: {sorted(r['test_items'])}\n")
            top5_pred = r['top_predicted'][:5] if len(r['top_predicted']) >= 5 else r['top_predicted']
            f.write(f"Top 5 predicted diseases: {top5_pred}\n\n")
        
        # Interesting findings
        patients_high_recall = sum(1 for r in valid_patients if r['recall'][k_idx] > 0.5)
        patients_3plus_hits = sum(1 for r in valid_patients if len(r['hits']) >= 3)
        patients_recall_1 = sum(1 for r in valid_patients if abs(r['recall'][k_idx] - 1.0) < 1e-6)
        
        f.write("=" * 60 + "\n")
        f.write("INTERESTING FINDINGS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Patients with recall = 1.0: {patients_recall_1}\n")
        f.write(f"Patients with recall > 0.5: {patients_high_recall}\n")
        f.write(f"Patients with 3+ hits: {patients_3plus_hits}\n\n")
        
        # Detailed analysis for patient with highest precision
        if len(sorted_by_precision) > 0:
            best_precision_patient = sorted_by_precision[0]
            f.write("=" * 80 + "\n")
            f.write("DETAILED ANALYSIS: PATIENT WITH HIGHEST PRECISION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Patient ID: {best_precision_patient['user_id']}\n")
            f.write(f"Precision@20: {best_precision_patient['precision'][k_idx]:.4f}\n")
            f.write(f"Recall@20: {best_precision_patient['recall'][k_idx]:.4f}\n")
            f.write(f"NDCG@20: {best_precision_patient['ndcg'][k_idx]:.4f}\n\n")
            
            f.write(f"Training diseases count: {len(best_precision_patient['training_items'])}\n")
            f.write(f"Testing diseases count: {len(best_precision_patient['test_items'])}\n\n")
            
            top10_pred = best_precision_patient['top_predicted'][:10] if len(best_precision_patient['top_predicted']) >= 10 else best_precision_patient['top_predicted']
            f.write(f"Top 10 predicted diseases: {top10_pred}\n")
            f.write(f"Hit diseases (correctly predicted): {sorted(best_precision_patient['hits'])}\n")
            f.write(f"All test diseases: {sorted(best_precision_patient['test_items'])}\n\n")
        
        # Detailed analysis for patient with highest recall
        if len(sorted_by_recall) > 0:
            best_recall_patient = sorted_by_recall[0]
            f.write("=" * 80 + "\n")
            f.write("DETAILED ANALYSIS: PATIENT WITH HIGHEST RECALL\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Patient ID: {best_recall_patient['user_id']}\n")
            f.write(f"Precision@20: {best_recall_patient['precision'][k_idx]:.4f}\n")
            f.write(f"Recall@20: {best_recall_patient['recall'][k_idx]:.4f}\n")
            f.write(f"NDCG@20: {best_recall_patient['ndcg'][k_idx]:.4f}\n\n")
            
            f.write(f"Training diseases count: {len(best_recall_patient['training_items'])}\n")
            f.write(f"Testing diseases count: {len(best_recall_patient['test_items'])}\n\n")
            
            top5_pred = best_recall_patient['top_predicted'][:5] if len(best_recall_patient['top_predicted']) >= 5 else best_recall_patient['top_predicted']
            f.write(f"Top 5 predicted diseases: {top5_pred}\n")
            f.write(f"Hit diseases (correctly predicted): {sorted(best_recall_patient['hits'])}\n")
            f.write(f"All test diseases: {sorted(best_recall_patient['test_items'])}\n\n")
    
    # Save CSV file
    import csv
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['user_id', 'precision', 'recall', 'ndcg', 'hit_ratio', 
                         'num_hits', 'num_test_diseases', 'num_training_diseases'])
        for r in sorted_by_precision:
            writer.writerow([
                r['user_id'],
                f"{r['precision'][k_idx]:.4f}",
                f"{r['recall'][k_idx]:.4f}",
                f"{r['ndcg'][k_idx]:.4f}",
                f"{r['hit_ratio'][k_idx]:.4f}",
                len(r['hits']),
                len(r['test_items']),
                len(r['training_items'])
            ])
    
    print(f"Detailed report saved to {output_file}")
    print(f"CSV file saved to {csv_file}")
