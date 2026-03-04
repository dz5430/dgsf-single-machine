# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# ---------- Load Product Info ----------
products_df = pd.read_csv("Facility_products.csv")  # Ensure this file is present

# ---------- Parameters ----------
num_instances_per_set = 100 # Number of scheduling instances to generate per set
min_batches = 30  # Minimum number of batches per instance
max_batches = 50  # Maximum number of batches per instance

# ---------- Configuration Selection ----------
configurations_to_generate = [
    {"name": "theta_max_6Itau", "theta_type": "max", "sigma_factor": "I*tau_avg"},
]

# ---------- Expand scalar to nx1 list ----------
def expand(scalar, n):
    return [float(scalar)] * n

# ---------- Generate Perfect Parallelogram ----------
def generate_perfect_parallelogram(I, tau_avg, theta, sigma):
    """
    Generate release and due times that form a perfect parallelogram
    before adding any fluctuations.
    """
    rho_perfect = np.zeros(I, dtype=float)
    eps_perfect = np.zeros(I, dtype=float)
    
    # Generate release times with perfect staggering
    if theta == np.pi/2:
        # Special case: all batches released at same time
        rho_perfect[:] = 0
    else:
        # Perfect stagger increment based on angle
        stagger_increment = 1 / np.tan(theta)
        for i in range(I):
            rho_perfect[i] = i * stagger_increment
    
    # Generate due times to form perfect parallelogram
    for i in range(I):
        eps_perfect[i] = rho_perfect[i] + sigma
    
    return rho_perfect, eps_perfect

# ---------- Add Fluctuations to Perfect Parallelogram ----------
def add_fluctuations(rho_perfect, eps_perfect, I, tau_avg):
    """
    Add controlled fluctuations to the perfect parallelogram.
    """
    # Calculate deviation parameters
    dev1 = tau_avg * I * 0.3 # Eps fluctuation
    dev2 =  0 # Rho fluctuation
    dev1_int = int(max(0,dev1))
    dev2_int = int(max(0,dev2))
    
    # Copy perfect arrays
    rho = rho_perfect.copy()
    eps = eps_perfect.copy()
    
    # Add fluctuations to release times
    for i in range(I):
        if i > 0:  # Keep first batch at t=0
            random_variation = np.random.randint(-dev2_int, dev2_int + 1)
            rho[i] = max(0, rho[i] + random_variation)
    
    # Add fluctuations to due times
    for i in range(I):
        random_variation = np.random.randint(-dev1_int, dev1_int + 1)
        eps[i] = eps[i] + random_variation
        # Ensure due time is reasonable (at least release time + small buffer)
        eps[i] = max(eps[i], rho[i] + tau_avg * 0.1)
    
    return rho.astype(int), eps.astype(int)

# ---------- Main Instance Generation Loop ----------
for config in configurations_to_generate:
    print(f"Generating {config['name']} instances...")
    data_list = []
    
    for instance_idx in range(num_instances_per_set):
        I = 100 # Number of batches per instance
    
        # Initialize arrays
        tau = np.zeros(I, dtype=int)
        
        # Without replacement
        # # --- Sample processing times WITHOUT replacement (compact) ---
        # tau = products_df.sample(n=I, replace=False)['processing_time'].to_numpy()
        # tau_avg = tau.mean()
        
        # Generate processing times first
        for i in range(I):
            pid = np.random.choice(products_df['product_id'])
            tau[i] = products_df.loc[products_df['product_id'] == pid, 'tau_m1'].values[0]
        
        tau_avg = np.mean(tau)
        
        # Calculate theta based on configuration
        theta_min = np.arctan(1/tau_avg)
        if config["theta_type"] == "min":
            theta = theta_min
        elif config["theta_type"] == "mid":
            # Instead of simple midpoint, use a more reasonable middle ground
            theta = theta_min + (np.pi/2 - theta_min) * 0.3
        else:  # "max"
            theta = np.pi/2
        
        # Calculate sigma (processing window width)
        sigma = tau_avg * I * 0.6
        
        # STEP 1: Generate perfect parallelogram
        rho_perfect, eps_perfect = generate_perfect_parallelogram(I, tau_avg, theta, sigma)
        
        # STEP 2: Add fluctuations to the perfect parallelogram
        rho, eps = add_fluctuations(rho_perfect, eps_perfect, I, tau_avg)
        
        # Reshape arrays for feature calculation
        rho = rho.reshape(-1, 1)
        tau = tau.reshape(-1, 1) 
        eps = eps.reshape(-1, 1)
        
        # Calculate slack
        slack = eps - rho - tau
        
        # Collect all selected features
        entry = {
            "I": I,
            "rho": rho.flatten().tolist(),
            "tau": tau.flatten().tolist(),
            "eps": eps.flatten().tolist(),
            "slack": slack.flatten().tolist(),
        }
        data_list.append(entry)
    
    # ---------- Save to Excel ----------
    df = pd.DataFrame(data_list)
    filename = f"Dev3_f5_singlemachine_instances_100_{config['name']}.xlsx"
    df.to_excel(filename, index=False)
    print(f"Saved {len(data_list)} instances to {filename}")
