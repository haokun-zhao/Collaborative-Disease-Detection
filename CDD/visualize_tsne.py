'''
t-SNE Visualization for Disease Embeddings
Created for CLDD model visualization
'''

import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import adjustText for automatic label adjustment
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not available. Using simple label adjustment algorithm.")

from CDD import CDD
from utility.helper import *
from utility.load_data import Data
from utility.parser import parse_args

def compute_comorbidity_degrees(plain_adj, n_users, n_items):
    """
    Compute comorbidity degrees for diseases from adjacency matrix
    The adjacency matrix is bipartite (users-diseases), so we compute
    disease-disease comorbidity through co-occurrence with users
    """
    # Extract user-disease interaction matrix R
    R = plain_adj[:n_users, n_users:].toarray()  # shape: (n_users, n_items)
    
    # Compute disease-disease comorbidity matrix: R^T * R
    # This gives the number of users who have both diseases
    disease_adj = R.T @ R  # shape: (n_items, n_items)
    
    # Set diagonal to 0 (disease with itself)
    np.fill_diagonal(disease_adj, 0)
    
    # Compute comorbidity degree for each disease (sum of co-occurrences)
    comorbidity_degrees = np.array(disease_adj.sum(axis=1)).flatten()
    
    # Also compute average comorbidity (mean of non-zero connections)
    non_zero_counts = (disease_adj > 0).sum(axis=1)
    avg_comorbidity = comorbidity_degrees / (non_zero_counts + 1e-10)
    print(f"Average comorbidity: {avg_comorbidity.mean():.4f}")
    
    return comorbidity_degrees, avg_comorbidity, disease_adj

def compute_discrepancy_pairs(disease_embeddings, disease_adj, top_k=20):
    """
    Compute top discrepancy pairs:
    - High comorbidity but far in embedding space (negative discrepancy)
    - Low comorbidity but close in embedding space (positive discrepancy)
    """
    n_diseases = disease_embeddings.shape[0]
    
    # Compute embedding similarities (cosine similarity)
    embedding_similarity = cosine_similarity(disease_embeddings)
    embedding_distance = 1 - embedding_similarity
    
    # Get comorbidity matrix
    comorbidity_matrix = disease_adj.toarray() if sp.issparse(disease_adj) else disease_adj
    
    # Normalize comorbidity for comparison (0-1 scale)
    max_comorbidity = comorbidity_matrix.max()
    if max_comorbidity > 0:
        normalized_comorbidity = comorbidity_matrix / max_comorbidity
    else:
        normalized_comorbidity = comorbidity_matrix
    
    # Find pairs with high discrepancy
    discrepancies = []
    
    for i in range(n_diseases):
        for j in range(i + 1, n_diseases):
            norm_comorbidity = normalized_comorbidity[i, j]
            emb_sim = embedding_similarity[i, j]
            emb_dist = embedding_distance[i, j]
            
            # Discrepancy metric:
            # Positive: low comorbidity but high similarity (CLDD found hidden relationship)
            # Negative: high comorbidity but low similarity (comorbidity doesn't match embedding)
            # We use: similarity - normalized_comorbidity
            # High positive: low comorbidity, high similarity (interesting case)
            # High negative: high comorbidity, low similarity (also interesting)
            discrepancy = emb_sim - norm_comorbidity
            
            discrepancies.append({
                'disease_i': i,
                'disease_j': j,
                'comorbidity': comorbidity_matrix[i, j],
                'normalized_comorbidity': norm_comorbidity,
                'embedding_distance': emb_dist,
                'embedding_similarity': emb_sim,
                'discrepancy': discrepancy
            })
    
    # Sort by absolute discrepancy (both positive and negative are interesting)
    discrepancies.sort(key=lambda x: abs(x['discrepancy']), reverse=True)
    
    return discrepancies[:top_k]

def get_icd_chapter(icd_code, icd_version=None):
    """
    Map ICD code to ICD-10 Chapter number based on ICD version
    Args:
        icd_code: ICD code (string or number)
        icd_version: ICD version (9 or 10), if None, will try to infer from code format
    Returns:
        chapter number (1-22) or 0 if unknown
    """
    if pd.isna(icd_code):
        return 0
    
    # Convert to string and remove any whitespace
    code_str = str(icd_code).strip().upper()
    
    # Determine ICD version if not provided
    if icd_version is None:
        # Try to infer from code format
        if code_str.isdigit() or (len(code_str) > 0 and code_str[0].isdigit()):
            icd_version = 9
        else:
            icd_version = 10
    
    # Handle ICD-9 codes
    if icd_version == 9:
        first_char = code_str[0]
        if first_char.isdigit():
            main3 = code_str[:3]
            try:
                code_num = int(main3)
            except ValueError:
                return None
        
            # ICD-9 to ICD-10 Chapter mapping based on ICD-9 ranges
            if 0 < code_num < 140:
                return 1  # Infectious diseases (001-139)
            elif code_num < 240:
                return 2  # Neoplasms (140-239)
            elif code_num < 280:
                return 3  # Blood diseases (240-279)
            elif code_num < 290:
                return 4  # Endocrine (280-289)
            elif code_num < 320:
                return 5  # Mental disorders (290-319)
            elif code_num < 360:
                return 6  # Nervous system (320-359)
            elif code_num < 380:
                return 7  # Eye (360-379)
            elif code_num < 390:
                return 8  # Ear (380-389)
            elif code_num < 460:
                return 9  # Circulatory (390-459)
            elif code_num < 520:
                return 10  # Respiratory (460-519)
            elif code_num < 580:
                return 11  # Digestive (520-579)
            elif code_num < 630:
                return 14  # Genitourinary (580-629)
            elif code_num < 680:
                return 14  # Genitourinary (630-679)
            elif code_num < 710:
                return 12  # Skin (680-709)
            elif code_num < 740:
                return 13  # Musculoskeletal (710-739)
            elif code_num < 760:
                return 13  # Musculoskeletal (740-759)
            elif code_num < 780:
                return 16  # Perinatal (760-779)
            elif code_num < 800:
                return 18  # Symptoms (780-799)
            elif code_num < 1000:
                return 19  # Injury (800-999)
            else:
                return 0
        
        if first_char == 'V':
            return 20
        if first_char == 'E':
            return 21
    
    # Handle ICD-10 codes
    elif icd_version == 10:
        if len(code_str) == 0:
            return 0
        
        first_char = code_str[0]
        
        # ICD-10 Chapter mapping
        if first_char == 'A' or first_char == 'B':
            return 1  # Certain infectious and parasitic diseases
        elif first_char == 'C' or (first_char == 'D' and len(code_str) > 1 and code_str[1] in '01234'):
            return 2  # Neoplasms (C00-D49)
        elif first_char == 'D' and len(code_str) > 1 and code_str[1] in '56789':
            return 3  # Diseases of blood and immune system (D50-D89)
        elif first_char == 'E':
            return 4  # Endocrine, nutritional and metabolic diseases
        elif first_char == 'F':
            return 5  # Mental and behavioural disorders
        elif first_char == 'G':
            return 6  # Diseases of the nervous system
        elif first_char == 'H' and len(code_str) > 1 and code_str[1] in '012345':
            return 7  # Diseases of the eye and adnexa (H00-H59)
        elif first_char == 'H' and len(code_str) > 1 and code_str[1] in '6789':
            return 8  # Diseases of the ear and mastoid process (H60-H95)
        elif first_char == 'I':
            return 9  # Diseases of the circulatory system
        elif first_char == 'J':
            return 10  # Diseases of the respiratory system
        elif first_char == 'K':
            return 11  # Diseases of the digestive system
        elif first_char == 'L':
            return 12  # Diseases of the skin and subcutaneous tissue
        elif first_char == 'M':
            return 13  # Diseases of the musculoskeletal system
        elif first_char == 'N':
            return 14  # Diseases of the genitourinary system
        elif first_char == 'O':
            return 15  # Pregnancy, childbirth and the puerperium
        elif first_char == 'P':
            return 16  # Certain conditions originating in the perinatal period
        elif first_char == 'Q':
            return 17  # Congenital malformations
        elif first_char == 'R':
            return 18  # Symptoms, signs and abnormal findings
        elif first_char == 'S' or first_char == 'T':
            return 19  # Injury, poisoning and external causes
        elif first_char == 'U':
            return 20  # Codes for special purposes
        elif first_char == 'V' or first_char == 'W' or first_char == 'X' or first_char == 'Y':
            return 21  # External causes of morbidity
        elif first_char == 'Z':
            return 22  # Factors influencing health status
        else:
            return 0  # Unknown
    
    return 0  # Unknown version

def load_icd_version_mapping(d_icd_diagnoses_csv_path):
    """
    Load ICD code to version mapping from d_icd_diagnoses.csv
    Returns a dictionary mapping ICD code (as string) to ICD version (9 or 10)
    """
    print(f"Loading ICD version mapping from {d_icd_diagnoses_csv_path}...")
    
    # Read the ICD diagnoses file
    df_icd = pd.read_csv(d_icd_diagnoses_csv_path)
    
    # Create mapping: ICD code -> ICD version
    # Convert ICD codes to string and remove leading zeros from the left
    icd_version_map = {}
    for _, row in df_icd.iterrows():
        # Convert to string and strip whitespace
        icd_code = str(row['icd_code']).strip()
        icd_version = int(row['icd_version'])
        icd_version_map[icd_code] = icd_version
    
    print(f"Loaded {len(icd_version_map)} ICD code mappings")
    version_counts = pd.Series(list(icd_version_map.values())).value_counts()
    print(f"ICD version distribution: {dict(version_counts)}")
    
    return icd_version_map

def load_ccsr_mapping(ccsr_mapping_csv_path):
    """
    Load ICD-10 to CCSR Category mapping from DXCCSR mapping file
    Returns a dictionary mapping ICD-10 code (normalized) to list of CCSR categories
    For codes with multiple categories, we'll use the default (Inpatient Default = 'Y' or first one)
    """
    print(f"Loading CCSR mapping from {ccsr_mapping_csv_path}...")
    
    # Read the CCSR mapping file
    df_ccsr = pd.read_csv(ccsr_mapping_csv_path, skiprows=1)  # Skip first row (description)
    
    # Create mapping: ICD-10 code -> CCSR Category
    # For codes with multiple categories, prefer Inpatient Default = 'Y'
    ccsr_mapping = {}
    ccsr_category_descriptions = {}
    
    for _, row in df_ccsr.iterrows():
        icd10_code = str(row['ICD-10-CM Code']).strip().upper()
        ccsr_category = str(row['CCSR Category']).strip()
        ccsr_description = str(row['CCSR Category Description']).strip()
        inpatient_default = str(row['Inpatient Default CCSR (Y/N/X)']).strip().upper()
        
        # Store category description
        ccsr_category_descriptions[ccsr_category] = ccsr_description
        
        # Normalize ICD-10 code (remove dots if any, ensure uppercase)
        icd10_normalized = icd10_code.replace('.', '')
        
        # If code not in mapping, add it
        if icd10_normalized not in ccsr_mapping:
            ccsr_mapping[icd10_normalized] = []
        
        # Add category with priority flag (Y = default)
        ccsr_mapping[icd10_normalized].append({
            'category': ccsr_category,
            'description': ccsr_description,
            'is_default': (inpatient_default == 'Y')
        })
    
    # For each code, select the default category (or first one if no default)
    ccsr_final_mapping = {}
    for icd10_code, categories in ccsr_mapping.items():
        # Find default category
        default_cat = None
        for cat_info in categories:
            if cat_info['is_default']:
                default_cat = cat_info['category']
                break
        
        # If no default, use first category
        if default_cat is None and len(categories) > 0:
            default_cat = categories[0]['category']
        
        ccsr_final_mapping[icd10_code] = default_cat
    
    print(f"Loaded {len(ccsr_final_mapping)} ICD-10 to CCSR mappings")
    print(f"Found {len(ccsr_category_descriptions)} unique CCSR categories")
    
    return ccsr_final_mapping, ccsr_category_descriptions

def load_icd9_to_icd10_mapping(gem_csv_path):
    """
    Load ICD-9 to ICD-10 mapping from icd9toicd10cmgem.csv
    Returns a dictionary mapping ICD-9 code to list of ICD-10 codes
    For codes with multiple mappings, we'll use the first one (or best match based on flags)
    """
    print(f"Loading ICD-9 to ICD-10 mapping from {gem_csv_path}...")
    
    # Read the GEM mapping file
    df_gem = pd.read_csv(gem_csv_path)
    
    # Create mapping: ICD-9 code -> ICD-10 code(s)
    icd9_to_icd10 = {}
    
    for _, row in df_gem.iterrows():
        icd9_code = str(row['icd9cm']).strip()
        icd10_code = str(row['icd10cm']).strip()
        approximate = int(row['approximate']) if pd.notna(row['approximate']) else 0
        no_map = int(row['no_map']) if pd.notna(row['no_map']) else 0
        
        # Skip if no mapping
        if no_map == 1:
            continue
        
        # Normalize ICD-9 code (remove dots, ensure uppercase)
        icd9_normalized = icd9_code.replace('.', '').upper()
        # Normalize ICD-10 code (remove dots, ensure uppercase)
        icd10_normalized = icd10_code.replace('.', '').upper()
        
        # If code not in mapping, add it
        if icd9_normalized not in icd9_to_icd10:
            icd9_to_icd10[icd9_normalized] = []
        
        # Add mapping with priority (non-approximate mappings first)
        icd9_to_icd10[icd9_normalized].append({
            'icd10': icd10_normalized,
            'approximate': approximate
        })
    
    # For each ICD-9 code, select the best ICD-10 mapping
    # Prefer non-approximate mappings, otherwise use first one
    icd9_to_icd10_final = {}
    for icd9_code, mappings in icd9_to_icd10.items():
        # Sort: non-approximate first
        mappings_sorted = sorted(mappings, key=lambda x: x['approximate'])
        # Use the first (best) mapping
        icd9_to_icd10_final[icd9_code] = mappings_sorted[0]['icd10']
    
    print(f"Loaded {len(icd9_to_icd10_final)} ICD-9 to ICD-10 mappings")
    
    return icd9_to_icd10_final

def build_structured_ccsr_labels(ccsr_categories):
    """
    Build structured labels based only on 3-letter prefix, ignoring numeric suffix.
    All categories with the same prefix will get the same label.
    
    Args:
        ccsr_categories: list of strings like ['CIR001', 'CIR002', 'RSP001', ...]
    Returns:
        category_to_label: dict mapping category to label (based on prefix)
        label_to_category: dict mapping label to representative category (first one with that prefix)
    """
    # Step 1: Extract unique prefixes and sort them
    unique_prefixes = sorted(set(cat[:3] for cat in ccsr_categories))
    
    # Step 2: Create prefix to label mapping
    prefix_to_label = {prefix: idx + 1 for idx, prefix in enumerate(unique_prefixes)}
    
    # Step 3: Map each category to its prefix's label
    category_to_label = {}
    label_to_category = {}
    
    for cat in ccsr_categories:
        prefix = cat[:3]
        label = prefix_to_label[prefix]
        category_to_label[cat] = label
        
        # Store the first category with this prefix as the representative
        if label not in label_to_category:
            label_to_category[label] = prefix  # Store prefix as representative
    
    return category_to_label, label_to_category

def load_ccsr_categories_from_csv(diag_adj_csv_path, n_items, ccsr_mapping=None, icd_version_map=None, icd9_to_icd10_map=None):
    """
    Load CCSR categories from diag_adj_with_ID.csv
    Processes both ICD-10 codes and ICD-9 codes (converts ICD-9 to ICD-10 first)
    Maps ICD-10 codes (original or converted from ICD-9) to CCSR Categories
    Uses icd_version_map to determine if a code is ICD-10 or ICD-9
    Uses icd9_to_icd10_map to convert ICD-9 codes to ICD-10
    Returns:
        ccsr_labels: array of CCSR category labels for each disease (0 for unmapped)
        disease_codes: list of original disease codes
        label_to_category: mapping from label to category prefix
        icd10_mapping: dict mapping original index to (original_code, icd10_code, is_converted)
    """
    print(f"Loading disease codes from {diag_adj_csv_path}...")
    # Read only the header row to get disease codes
    df_header = pd.read_csv(diag_adj_csv_path, nrows=0)
    disease_codes = df_header.columns[1:].tolist()  # Skip first column (subject_id)
    
    # Ensure we have the right number of diseases
    if len(disease_codes) != n_items:
        print(f"Warning: Number of disease codes ({len(disease_codes)}) doesn't match n_items ({n_items})")
        disease_codes = disease_codes[:n_items]
    
    # First pass: collect all CCSR categories that appear in the data
    ccsr_labels = []
    matched_count = 0
    icd9_count = 0
    icd9_converted_count = 0
    unmatched_codes = []
    categories_in_data = set()
    icd10_mapping = {}  # Store mapping: index -> (original_code, icd10_code, is_converted)
    
    for idx, code in enumerate(disease_codes):
        code_str = str(code).strip()
        code_normalized = code_str.replace('.', '').upper()
        
        # Check if it's ICD-10 using icd_version_map
        is_icd10 = False
        icd10_code_to_use = None
        
        if icd_version_map is not None:
            # Try exact match first
            if code_str in icd_version_map:
                icd_version = icd_version_map[code_str]
                is_icd10 = (icd_version == 10)
            else:
                # Try normalized match (remove dots, uppercase)
                if code_normalized in icd_version_map:
                    icd_version = icd_version_map[code_normalized]
                    is_icd10 = (icd_version == 10)
                else:
                    # Code not found in mapping - cannot determine version
                    ccsr_labels.append(None)
                    unmatched_codes.append(code_str)
                    continue
        else:
            # No version map provided - cannot determine version reliably
            ccsr_labels.append(None)
            unmatched_codes.append(code_str)
            continue
        
        if is_icd10:
            # Original ICD-10 code
            icd10_code_to_use = code_normalized
            is_converted = False
        else:
            # ICD-9 code, try to convert to ICD-10
            icd9_count += 1
            if icd9_to_icd10_map is not None and code_normalized in icd9_to_icd10_map:
                icd10_code_to_use = icd9_to_icd10_map[code_normalized]
                icd9_converted_count += 1
                is_converted = True
            else:
                # Cannot convert ICD-9 to ICD-10
                ccsr_labels.append(None)
                unmatched_codes.append(code_str)
                continue
        
        # Store mapping information
        icd10_mapping[idx] = (code_str, icd10_code_to_use, is_converted)
        
        # Look up CCSR category using ICD-10 code (original or converted)
        if icd10_code_to_use is not None and ccsr_mapping is not None and icd10_code_to_use in ccsr_mapping:
            ccsr_category = ccsr_mapping[icd10_code_to_use]
            if ccsr_category is not None:
                ccsr_labels.append(ccsr_category)
                categories_in_data.add(ccsr_category)
                matched_count += 1
            else:
                ccsr_labels.append(None)
                unmatched_codes.append(code_str)
        else:
            ccsr_labels.append(None)
            unmatched_codes.append(code_str)
    
    # Create mapping from CCSR category code to numeric label (only for categories in data)
    # Use structured sorting: group by 3-letter prefix, then sort by number within each group
    category_to_label, label_to_category = build_structured_ccsr_labels(list(categories_in_data))
    
    # Convert category names to numeric labels
    ccsr_labels = np.array([category_to_label.get(cat, 0) if cat is not None else 0 for cat in ccsr_labels])
    
    print(f"Loaded {len(disease_codes)} disease codes")
    print(f"ICD-10 codes matched to CCSR: {matched_count}")
    print(f"ICD-9 codes found: {icd9_count}")
    print(f"ICD-9 codes converted to ICD-10: {icd9_converted_count}")
    print(f"ICD-9 codes not converted: {icd9_count - icd9_converted_count}")
    print(f"Unmatched codes (not in version map or not in CCSR mapping): {len(unmatched_codes)}")
    if len(unmatched_codes) > 0 and len(unmatched_codes) <= 10:
        print(f"Unmatched codes (first 10): {unmatched_codes[:10]}")
    elif len(unmatched_codes) > 10:
        print(f"Unmatched codes: {len(unmatched_codes)} codes (showing first 10: {unmatched_codes[:10]})")
    
    # Print CCSR category distribution
    unique_labels = np.unique(ccsr_labels[ccsr_labels > 0])
    if len(unique_labels) > 0:
        category_counts = {label_to_category.get(label, f"Unknown_{label}"): count 
                          for label, count in zip(unique_labels, np.bincount(ccsr_labels[ccsr_labels > 0])[unique_labels])}
        print(f"CCSR category distribution: {dict(list(category_counts.items()))}")
    
    return ccsr_labels, disease_codes, label_to_category, icd10_mapping

def load_icd_categories_from_csv(diag_adj_csv_path, n_items, icd_version_map=None):
    """
    Load ICD categories from diag_adj_with_ID.csv
    The disease codes are in columns starting from index 1 (B column)
    Uses d_icd_diagnoses.csv to determine ICD version for each code
    Returns array of chapter numbers for each disease
    """
    print(f"Loading disease codes from {diag_adj_csv_path}...")
    # Read only the header row to get disease codes
    df_header = pd.read_csv(diag_adj_csv_path, nrows=0)
    disease_codes = df_header.columns[1:].tolist()  # Skip first column (subject_id)
    
    # Ensure we have the right number of diseases
    if len(disease_codes) != n_items:
        print(f"Warning: Number of disease codes ({len(disease_codes)}) doesn't match n_items ({n_items})")
        disease_codes = disease_codes[:n_items]
    
    # Map each disease code to its ICD chapter using version information
    chapter_labels = []
    matched_count = 0
    unmatched_codes = []
    
    for code in disease_codes:
        code_str = str(code).strip()
        
        # Look up ICD version from mapping
        icd_version = None
        if icd_version_map is not None:
            # Try normalized code match (since mapping also uses normalized codes)
            if code_str in icd_version_map:
                icd_version = icd_version_map[code_str]
                matched_count += 1
            else:
                unmatched_codes.append(code_str)
        
        # Get chapter based on code and version
        chapter = get_icd_chapter(code, icd_version)
        chapter_labels.append(chapter)
    
    chapter_labels = np.array(chapter_labels)
    
    print(f"Loaded {len(disease_codes)} disease codes")
    if icd_version_map is not None:
        print(f"Matched {matched_count}/{len(disease_codes)} codes with version information")
        if len(unmatched_codes) > 0 and len(unmatched_codes) <= 10:
            print(f"Unmatched codes (first 10): {unmatched_codes[:10]}")
        elif len(unmatched_codes) > 10:
            print(f"Unmatched codes: {len(unmatched_codes)} codes (showing first 10: {unmatched_codes[:10]})")
    
    # Print chapter distribution
    unique_chapters = np.unique(chapter_labels[chapter_labels > 0])
    print(f"Chapter distribution: {dict(zip(unique_chapters, np.bincount(chapter_labels[chapter_labels > 0])[unique_chapters]))}")
    
    return chapter_labels, disease_codes

def adjust_label_positions(xy_points, text_objects, ax, min_distance=2.0, max_iter=100):
    """
    Simple algorithm to adjust label positions to avoid overlaps
    Uses a force-based approach to push overlapping labels apart
    """
    if len(text_objects) == 0:
        return
    
    # Get initial text positions (relative to points)
    text_positions = []
    for i, (point, text_obj) in enumerate(zip(xy_points, text_objects)):
        pos = text_obj.get_position()
        # Store offset from point
        offset_x = pos[0] - point[0]
        offset_y = pos[1] - point[1]
        text_positions.append([offset_x, offset_y])
    
    text_positions = np.array(text_positions)
    xy_points = np.array(xy_points)
    
    # Iterative adjustment
    for iteration in range(max_iter):
        forces = np.zeros_like(text_positions)
        
        # Calculate repulsive forces between nearby labels
        for i in range(len(text_objects)):
            for j in range(i + 1, len(text_objects)):
                # Current text positions
                pos_i = xy_points[i] + text_positions[i]
                pos_j = xy_points[j] + text_positions[j]
                
                # Distance between text positions
                dx = pos_j[0] - pos_i[0]
                dy = pos_j[1] - pos_i[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < min_distance and dist > 1e-6:
                    # Apply repulsive force
                    force_magnitude = (min_distance - dist) / dist
                    force_x = dx * force_magnitude * 0.1
                    force_y = dy * force_magnitude * 0.1
                    
                    forces[i] -= [force_x, force_y]
                    forces[j] += [force_x, force_y]
        
        # Apply forces with damping
        text_positions += forces * 0.5
        
        # Limit maximum offset from point
        max_offset = 5.0
        for i in range(len(text_positions)):
            offset_mag = np.sqrt(text_positions[i][0]**2 + text_positions[i][1]**2)
            if offset_mag > max_offset:
                text_positions[i] = text_positions[i] / offset_mag * max_offset
        
        # Update text positions
        for i, text_obj in enumerate(text_objects):
            new_x = xy_points[i][0] + text_positions[i][0]
            new_y = xy_points[i][1] + text_positions[i][1]
            text_obj.set_position((new_x, new_y))
        
        # Check if forces are small enough
        if np.max(np.abs(forces)) < 0.01:
            break

def annotate_diseases(ax, tsne_x, tsne_y, disease_codes, target_codes, labels, colors=None, text_color='black'):
    """
    Annotate specific diseases on t-SNE plot with automatic label adjustment to avoid overlaps
    
    Args:
        ax: matplotlib axis object
        tsne_x: x coordinates of t-SNE results (filtered, length = len(disease_codes))
        tsne_y: y coordinates of t-SNE results (filtered, length = len(disease_codes))
        disease_codes: list of disease codes (filtered, only ICD-10 codes)
        target_codes: ICD codes to annotate, like ['I10','E11','J44']
        labels: readable labels ['Hypertension','Type 2 DM','COPD']
        colors: list of colors for each disease (same length as target_codes), or None for default red
        text_color: text color for annotations
    """
    # Normalize codes (remove dot, uppercase)
    normalized_codes = [str(c).replace('.', '').upper() for c in disease_codes]
    
    # Default color if not provided
    if colors is None:
        colors = ['red'] * len(target_codes)
    
    # Collect points and text objects
    points = []
    text_objects = []
    scatter_objects = []
    
    # Use global median to decide default offset direction
    x_center = np.median(tsne_x) if len(tsne_x) > 0 else 0.0
    y_center = np.median(tsne_y) if len(tsne_y) > 0 else 0.0
    offset = 2.0

    for icd, label, color in zip(target_codes, labels, colors):
        icd_norm = str(icd).replace('.', '').upper()
        
        if icd_norm in normalized_codes:
            idx = normalized_codes.index(icd_norm)
            x, y = tsne_x[idx], tsne_y[idx]
            points.append((x, y))
            
            # Draw scatter point
            scatter = ax.scatter([x], [y], color=color, s=120, edgecolors='white', linewidth=1.5, zorder=10)
            scatter_objects.append(scatter)
            
            # Determine default offset direction so label points away from plot center
            dx = offset if x >= x_center else -offset
            dy = offset if y >= y_center else -offset

            # Create text annotation
            text_obj = ax.text(
                x + dx, y + dy, 
                label, 
                fontsize=10, color=text_color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color, linewidth=1.0)
            )
            text_objects.append(text_obj)
        else:
            print(f"[Warning] ICD code {icd} not found in filtered disease codes!")
    
    # Adjust text positions to avoid overlaps
    if HAS_ADJUSTTEXT and len(text_objects) > 0:
        # Use adjustText library if available
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        adjust_text(
            text_objects,
            x_coords,
            y_coords,
            ax=ax,
            expand_points=(2, 2),
            expand_text=(2, 2),
            # arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.0, shrinkA=5, shrinkB=0)
        )
    elif len(text_objects) > 0:
        # Use simple adjustment algorithm
        adjust_label_positions(points, text_objects, ax)

def visualize_tsne(disease_embeddings, avg_comorbidity, category_labels=None, 
                   discrepancy_pairs=None, perplexity=30, learning_rate=200, 
                   random_state=42, save_path='tsne_visualization.png', 
                   label_to_category=None, use_ccsr=False, disease_codes=None,
                   annotate_targets=None, annotate_labels=None, annotate_colors=None):
    """
    Create t-SNE visualization with three subplots:
    (a) Diseases colored by ICD/CCS category or CCSR Category
    (b) Diseases colored/sized by comorbidity degree
    (c) Top discrepancy pairs highlighted
    
    Args:
        use_ccsr: If True, category_labels are CCSR category labels, else ICD chapter labels
        label_to_category: Mapping from numeric label to category name (for CCSR)
    """
    print(f"Running t-SNE with perplexity={perplexity}, learning_rate={learning_rate}...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, 
                random_state=random_state, n_iter=1000, verbose=0)
    tsne_results = tsne.fit_transform(disease_embeddings)
    
    # Filter to only show ICD-10 diseases (category_labels > 0)
    if category_labels is not None:
        icd10_mask = category_labels > 0
        icd10_indices = np.where(icd10_mask)[0]
        tsne_results_icd10 = tsne_results[icd10_indices]
        category_labels_icd10 = category_labels[icd10_indices]
        avg_comorbidity_icd10 = avg_comorbidity[icd10_indices]
        # Create mapping from original index to filtered index for annotation
        original_to_filtered = {orig_idx: filtered_idx for filtered_idx, orig_idx in enumerate(icd10_indices)}
        # Filter disease codes to match filtered indices
        if disease_codes is not None:
            disease_codes_filtered = [disease_codes[i] for i in icd10_indices]
        else:
            disease_codes_filtered = None
        print(f"Filtering: {len(icd10_indices)} ICD-10 diseases out of {len(disease_embeddings)} total")
    else:
        icd10_mask = np.ones(len(disease_embeddings), dtype=bool)
        icd10_indices = np.arange(len(disease_embeddings))
        tsne_results_icd10 = tsne_results
        category_labels_icd10 = category_labels
        avg_comorbidity_icd10 = avg_comorbidity
        original_to_filtered = {i: i for i in range(len(disease_embeddings))}
        disease_codes_filtered = disease_codes
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 6))
    
    # Subplot (a): Diseases colored by category
    ax1 = plt.subplot(1, 2, 1)
    if category_labels_icd10 is not None and len(np.unique(category_labels_icd10)) > 1:
        unique_labels = np.unique(category_labels_icd10)
        num_categories = len(unique_labels)
        
        if use_ccsr:
            # For CCSR categories, use a colormap that can handle many categories
            # Use 'tab20' for up to 20 categories, 'Set3' for more
            if num_categories <= 20:
                cmap = 'tab20'
                vmax = 20
            elif num_categories <= 40:
                cmap = 'Set3'
                vmax = num_categories
            else:
                # For many categories, use a continuous colormap
                cmap = 'nipy_spectral'
                vmax = num_categories
            
            scatter1 = ax1.scatter(tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
                                  c=category_labels_icd10, cmap=cmap, alpha=0.6, s=20, 
                                  vmin=1, vmax=vmax)
            ax1.set_title('Diseases by CCSR Category', fontsize=14, fontweight='bold')
            cbar1 = plt.colorbar(scatter1, ax=ax1, label='CCSR Category')
            
            # For CCSR, we can't show all category names on colorbar if there are many
            if num_categories <= 20 and label_to_category is not None:
                # Show category codes on colorbar
                tick_labels = [label_to_category.get(label, f"Cat_{label}") 
                              for label in unique_labels[:20]]
                cbar1.set_ticks(unique_labels[:20])
                cbar1.set_ticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        else:
            # Use tab20 colormap for ICD chapters
            scatter1 = ax1.scatter(tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
                                  c=category_labels_icd10, cmap='tab20', alpha=0.6, s=20, 
                                  vmin=1, vmax=22)
            ax1.set_title('Diseases by ICD Chapter', fontsize=14, fontweight='bold')
            cbar1 = plt.colorbar(scatter1, ax=ax1, label='ICD Chapter')
            # Set colorbar ticks to show chapter numbers
            cbar1.set_ticks(unique_labels)
    else:
        ax1.scatter(tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], alpha=0.6, s=20, c='gray')
        title = 'Diseases by CCSR Category' if use_ccsr else 'Diseases by ICD Chapter'
        ax1.set_title(f'{title}\n(No category data available)', 
                     fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations if provided
    if annotate_targets is not None and annotate_labels is not None and disease_codes_filtered is not None:
        annotate_diseases(ax1, tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
                         disease_codes_filtered, annotate_targets, annotate_labels, 
                         colors=annotate_colors)
    
    # Subplot (b): Diseases colored/sized by comorbidity degree
    ax2 = plt.subplot(1, 2, 2)
    
    # Apply log transformation to improve color distinction for small values
    # Add 1 to avoid log(0) and handle small values
    log_avg_comorbidity = np.log1p(avg_comorbidity_icd10)  # log1p(x) = log(1 + x)
    
    # Normalize log-transformed values for size scaling
    normalized_degrees = (log_avg_comorbidity - log_avg_comorbidity.min()) / \
                        (log_avg_comorbidity.max() - log_avg_comorbidity.min() + 1e-10)
    
    # Use high-saturation colormap for comorbidity degree
    # 'plasma', 'inferno', 'magma', 'turbo' are high-saturation options
    scatter2 = ax2.scatter(tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
                          c=log_avg_comorbidity, cmap='viridis', 
                          s=20 + normalized_degrees * 100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_title('Diseases by Comorbidity Degree', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Comorbidity Degree')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations if provided
    if annotate_targets is not None and annotate_labels is not None and disease_codes_filtered is not None:
        annotate_diseases(ax2, tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
                         disease_codes_filtered, annotate_targets, annotate_labels,
                         colors=annotate_colors)
    
    # Subplot (c): Top discrepancy pairs
    # ax3 = plt.subplot(1, 3, 3)
    # # Plot all ICD-10 diseases
    # ax3.scatter(tsne_results_icd10[:, 0], tsne_results_icd10[:, 1], 
    #             alpha=0.3, s=10, c='lightgray', label='All ICD-10 diseases')
    
    # if discrepancy_pairs is not None and len(discrepancy_pairs) > 0:
    #     # Create mapping from original index to ICD-10 filtered index
    #     index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(icd10_indices)}
        
    #     # Highlight top discrepancy pairs (only for ICD-10 diseases)
    #     highlighted_diseases = set()
    #     valid_pairs = 0
    #     for pair in discrepancy_pairs[:10]:  # Show top 10 pairs
    #         i, j = pair['disease_i'], pair['disease_j']
    #         # Only show pairs where both diseases are ICD-10
    #         if i in index_mapping and j in index_mapping:
    #             i_icd10 = index_mapping[i]
    #             j_icd10 = index_mapping[j]
    #             highlighted_diseases.add(i_icd10)
    #             highlighted_diseases.add(j_icd10)
                
    #             # Draw line between pair
    #             ax3.plot([tsne_results_icd10[i_icd10, 0], tsne_results_icd10[j_icd10, 0]], 
    #                     [tsne_results_icd10[i_icd10, 1], tsne_results_icd10[j_icd10, 1]], 
    #                     'r-', alpha=0.5, linewidth=1)
    #             valid_pairs += 1
        
    #     # Highlight disease points
    #     if len(highlighted_diseases) > 0:
    #         highlighted_indices = list(highlighted_diseases)
    #         ax3.scatter(tsne_results_icd10[highlighted_indices, 0], 
    #                 tsne_results_icd10[highlighted_indices, 1], 
    #                 c='red', s=50, alpha=0.8, edgecolors='black', linewidth=1.5,
    #                 label='Discrepancy pairs', zorder=5)
        
    #     ax3.set_title(f'(c) Top Discrepancy Pairs\n({valid_pairs} pairs shown)', 
    #                 fontsize=14, fontweight='bold')
    #     ax3.legend(fontsize=10)
    # else:
    #     ax3.set_title('(c) Top Discrepancy Pairs\n(No pairs computed)', 
    #                 fontsize=14, fontweight='bold')
    
    # ax3.set_xlabel('t-SNE Dimension 1', fontsize=12)
    # ax3.set_ylabel('t-SNE Dimension 2', fontsize=12)
    # ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

def extract_disease_embeddings(model, n_items):
    """
    Extract final layer disease embeddings (z_d^(L)) from CDD model
    """
    model.eval()
    with torch.no_grad():
        # Create dummy inputs to trigger forward pass
        dummy_users = torch.tensor([0], dtype=torch.long).to(model.device)
        dummy_pos_items = torch.tensor(list(range(n_items)), dtype=torch.long).to(model.device)
        dummy_neg_items = torch.tensor([], dtype=torch.long).to(model.device)
        
        # Forward pass to get embeddings
        _, _, _ = model(dummy_users, dummy_pos_items, dummy_neg_items, drop_flag=False)
        
        # Extract final disease embeddings (z_d)
        disease_embeddings = model.final_item_embeddings.cpu().numpy()
    
    return disease_embeddings

def main():
    data_generator = Data(path='../Data/mimicIV', batch_size=1024)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    
    # Check if disease embeddings file exists
    disease_emb_final_path = 'disease_emb_final2.npy'
    if os.path.exists(disease_emb_final_path):
        print(f"Loading disease embeddings from {disease_emb_final_path}...")
        disease_embeddings = np.load(disease_emb_final_path)
        print(f"Disease embeddings shape: {disease_embeddings.shape}")
        
        # Verify shape matches expected number of items
        if disease_embeddings.shape[0] != data_generator.n_items:
            print(f"Warning: Loaded embeddings shape {disease_embeddings.shape[0]} doesn't match n_items {data_generator.n_items}")
            print("Re-extracting disease embeddings from model...")
            # Need to generate embeddings from model
            disease_embeddings = None  # Will be generated below
        else:
            print("Disease embeddings loaded successfully!")
    else:
        print(f"Disease embeddings file not found at {disease_emb_final_path}")
        print("Will extract embeddings from CDD model...")
        disease_embeddings = None  # Will be generated below
    
    # Generate embeddings from model if needed
    if disease_embeddings is None:
        print("\nInitializing CDD model to extract embeddings...")
        # Parse arguments
        args = parse_args()
        args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        
        # Load feature matrix
        feature_path = os.path.join(args.data_path, args.dataset, 'feature.npz')
        if not os.path.exists(feature_path):
            feature_path = '../Data/mimicIV/feature.npz'  # fallback path
        feature_matrix = sp.load_npz(feature_path)
        
        args.node_dropout = eval(args.node_dropout)
        args.mess_dropout = eval(args.mess_dropout)
        
        # Initialize and load model
        model = CDD(data_generator.n_users,
                     data_generator.n_items,
                     norm_adj,
                     feature_matrix,
                     args)
        model.to(args.device)
        
        # Load checkpoint
        checkpoint_path = os.path.join(args.weights_path, '449.pkl')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Cannot extract embeddings without trained model. Exiting.")
            return
        
        # Extract disease embeddings
        print("Extracting disease embeddings from model...")
        disease_embeddings = extract_disease_embeddings(model, data_generator.n_items)
        print(f"Disease embeddings shape: {disease_embeddings.shape}")
        
        # Save embeddings
        print(f"Saving disease embeddings to {disease_emb_final_path}...")
        np.save(disease_emb_final_path, disease_embeddings)
        print("Disease embeddings saved successfully!")
    
    print("Computing comorbidity degrees...")
    # Compute comorbidity degrees
    comorbidity_degrees, avg_comorbidity, disease_adj = compute_comorbidity_degrees(
        plain_adj, data_generator.n_users, data_generator.n_items)
    print(f"Comorbidity degrees range: [{comorbidity_degrees.min():.2f}, {comorbidity_degrees.max():.2f}]")
    
    print("Loading ICD version mapping...")
    # Load ICD version mapping from d_icd_diagnoses.csv
    d_icd_diagnoses_csv_path = os.path.join('d_icd_diagnoses.csv')
    icd_version_map = load_icd_version_mapping(d_icd_diagnoses_csv_path)
    
    print("Loading ICD-9 to ICD-10 mapping...")
    # Load ICD-9 to ICD-10 mapping from GEM file
    gem_csv_path = os.path.join('icd9toicd10cmgem.csv')
    icd9_to_icd10_map = load_icd9_to_icd10_mapping(gem_csv_path)
    
    print("Loading CCSR categories...")
    # Load CCSR mapping from DXCCSR file
    ccsr_mapping_csv_path = os.path.join('DXCCSR-Reference-File-v2025-1(DX_to_CCSR_Mapping).csv')
    
    ccsr_mapping, ccsr_category_descriptions = load_ccsr_mapping(ccsr_mapping_csv_path)
    
    # Load CCSR categories from CSV file
    # Processes both ICD-10 codes and ICD-9 codes (converts ICD-9 to ICD-10 first)
    diag_adj_csv_path = os.path.join('diag_adj_with_ID.csv')
    
    category_labels, disease_codes, label_to_category, icd10_mapping = load_ccsr_categories_from_csv(
        diag_adj_csv_path, data_generator.n_items, ccsr_mapping, icd_version_map, icd9_to_icd10_map)
    print(f"Number of CCSR categories: {len(np.unique(category_labels[category_labels > 0]))}")
    
    # Export all ICD-10 codes (including converted from ICD-9) to file
    # print("\nExporting all ICD-10 codes to file...")
    # icd10_export_data = []
    # for idx, (original_code, icd10_code, is_converted) in icd10_mapping.items():
    #     if category_labels[idx] > 0:  # Only export codes that have CCSR categories
    #         icd10_export_data.append({
    #             'index': idx,
    #             'original_code': original_code,
    #             'icd10_code': icd10_code,
    #             'is_converted_from_icd9': is_converted,
    #             'ccsr_label': category_labels[idx],
    #             'ccsr_category': label_to_category.get(category_labels[idx], 'Unknown')
    #         })
    
    # # Export to CSV file
    # icd10_export_path = 'icd10_codes_all.csv'
    # df_icd10 = pd.DataFrame(icd10_export_data)
    # df_icd10 = df_icd10.sort_values('icd10_code')
    # df_icd10.to_csv(icd10_export_path, index=False)
    # print(f"Exported {len(icd10_export_data)} ICD-10 codes to {icd10_export_path}")
    
    # # Also export as simple text file (one code per line)
    # icd10_txt_path = 'icd10_codes_all.txt'
    # with open(icd10_txt_path, 'w', encoding='utf-8') as f:
    #     for row in sorted(icd10_export_data, key=lambda x: x['icd10_code']):
    #         f.write(f"{row['icd10_code']}\n")
    # print(f"Exported {len(icd10_export_data)} ICD-10 codes to {icd10_txt_path}")
    
    # # Export conversion mapping (ICD-9 to ICD-10)
    # conversion_data = [row for row in icd10_export_data if row['is_converted_from_icd9']]
    # if len(conversion_data) > 0:
    #     conversion_path = 'icd9_to_icd10_conversion.csv'
    #     df_conversion = pd.DataFrame(conversion_data)
    #     df_conversion = df_conversion.sort_values('original_code')
    #     df_conversion.to_csv(conversion_path, index=False)
    #     print(f"Exported {len(conversion_data)} ICD-9 to ICD-10 conversions to {conversion_path}")
    
    print("Computing discrepancy pairs...")
    # Compute discrepancy pairs
    discrepancy_pairs = compute_discrepancy_pairs(disease_embeddings, disease_adj, top_k=20)
    print(f"Found {len(discrepancy_pairs)} discrepancy pairs")
    
    # Print some example discrepancy pairs
    print("\nTop 5 discrepancy pairs:")
    for i, pair in enumerate(discrepancy_pairs[:5]):
        print(f"  {i+1}. Diseases {pair['disease_i']}-{pair['disease_j']}: "
              f"comorbidity={pair['comorbidity']:.4f}, "
              f"embedding_sim={pair['embedding_similarity']:.4f}, "
              f"discrepancy={pair['discrepancy']:.4f}")
    
    # Define target diseases to annotate (grouped by category)
    target_icd_groups = [
        # Cardiometabolic
        ['I10', 'I110', 'I120', 'I2510'],#'I5021', 'I5031', 
         #'E780', 'E781',],# 'E6601'],
        # Respiratory / Infectious
        [#'J441', 
         'J449', 'J9600', 'J9601', 'J690'],
        # Neuro / Psych
        ['G931'], #'G9340', 'G459',
         #'F05', 'F329', 
        ['F1010', 'F1020'],
        # Digestive
        # ['K210', 'K219', 'K254', 'K259', 'K7030', 'K7040'],
    ]
    
    target_labels_groups = [
        # Cardiometabolic
        ['Hypertension', 'Hypertensive heart disease w/ heart failure', 'HTN CKD', 
         'Chronic IHD'], #'Systolic HF', 'Diastolic HF',
         #'Hypercholesterolemia', 'Hyperglyceridemia'],#, 'Obesity'],
        # Respiratory/Infectious
        [#'COPD with (acute) exacerbation', 
         'COPD', 'Resp failure', 'Resp failure w/ hypoxia',
         'Aspiration pneumonia'],
        # Neuro/Psych
        ['Anoxic brain damage'], #'Encephalopathy', 'TIA',
        #  'Delirium', 'Depression', 
        ['Alcohol abuse', 'Alcohol dep'],
        # Digestive
        # ['GERD w/ esophagitis', 'GERD w/o esophagitis', 'Gastric ulcer', 'Peptic ulcer', 
        #  'Alcoholic liver disease', 'Cirrhosis'],
    ]
    
    # Flatten the groups into single lists
    target_icd = [code for group in target_icd_groups for code in group]
    target_labels = [label for group in target_labels_groups for label in group]
    
    # Define colors for each category (high saturation colors)
    category_colors = ['#ceddf1',
                      '#e9e9ba',   # Blue for Respiratory/Infectious
                      '#dddddd',   # Purple for Neuro/Psych
                      '#eeadda']   # Orange for Digestive
    
    # Create color list for each disease based on its category
    annotate_colors = []
    for i, group in enumerate(target_icd_groups):
        annotate_colors.extend([category_colors[i]] * len(group))
    
    print("\nGenerating visualizations...")
    # Generate visualizations with different perplexity values
    for perplexity in [50]:
        for learning_rate in [200]:
            save_path = f'tsne_visualization_ccsr_p{perplexity}_lr{learning_rate}.png'
            visualize_tsne(disease_embeddings, avg_comorbidity, category_labels,
                          discrepancy_pairs, perplexity=perplexity, 
                          learning_rate=learning_rate, random_state=42,
                          save_path=save_path, use_ccsr=True, 
                          label_to_category=label_to_category,
                          disease_codes=disease_codes,
                          annotate_targets=target_icd,
                          annotate_labels=target_labels,
                          annotate_colors=annotate_colors)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()

