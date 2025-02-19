# The Missing Role of Negative Feedback in DIME

This repository contains modified scripts based on the original work from the [DIME-SIGIR-2024 repository](https://github.com/guglielmof/DIME-SIGIR-2024/tree/main). Our modifications for reproducibility and replicability.

## Repository Structure

- **`dimension_filters/`**  
  Contains the Dimension Importance Estimation (DIME) functions used to compute the importance score. The code in this folder inherits from the `Abstract_Filter` class, which sets up the operative functionality of the filters.

The filters used to replicate the DIME in Section 4 are:
  - **PRFEclipse**: Builds a filter based on pseudo-relevance feedback (PRF) and pseudo-irrelevant feedback.
  - **LLMEclipse**: Builds a filter based on LLM-generated answers and pseudo-irrelevant feedback.


- **`memmap_interface/`**  
  Includes the code for building the memmap data structure used to store the embeddings of various collections and query sets.

- **`searching/`**  
  Contains two key scripts:
  - **`build_index`**: Computes the FAISS index given the memmap data.
  - **`search_faiss`**: Implements the retrieval pipeline without the DIME. This script outputs a file containing the retrieved documents for every query in the collection.

The **grid_search.ipynb** store the hyperparameters we used to grid search for the Section 4.

## Main Scripts

- **`main.py`**  
  Executes the DIME pipeline using the provided configuration.

- **`main_active_feedback.py`**  
  A variant of the main script, this version includes the Active Feedback functionality.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/giulio-derasmo/The-Missing-Role-of-Negative-Feedback-in-DIME.git
   cd .
   ```

2. **Set up your environment and install the required dependencies.**
3. **Run the main pipeline:** (example)
   ```bash
   python rq1.py -c deeplearning20 -r tasb -f TopkFilter --kpos 1 --hyperparams_filename No 
   ```
