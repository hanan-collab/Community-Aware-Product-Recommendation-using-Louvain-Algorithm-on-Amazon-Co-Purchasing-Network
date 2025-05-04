# Product Co-Purchase Network Analysis

This project analyzes product co-purchasing patterns using community detection and network analysis.

## Requirements

- Python 3.7+
- The following Python packages (see [`requirements.txt`](requirements.txt)):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - networkx

## Dataset

You need two CSV files in this directory:

- **copurchase.csv**  
  Contains product co-purchase relationships.  
  **Required columns:**  
  - `Source`: Product ID of the source product  
  - `Target`: Product ID of the target product  
  - `weight` (optional): Strength or frequency of co-purchase

- **products.csv**  
  Contains product metadata.  
  **Required columns:**  
  - `id` or `product_id`: Unique identifier for each product  
  - Other columns (e.g., `title`, `category`, `price`) are optional but recommended

**Note:**  
- All product IDs in `Source` and `Target` columns of `copurchase.csv` should exist in the `id` or `product_id` column of `products.csv`.

## Setup

1. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

2. **Prepare your data:**

   - Ensure `copurchase.csv` and `products.csv` are present in this directory.

3. **Run the analysis:**

   ```
   python louvain.py
   ```

   This will:
   - Analyze the co-purchase network.
   - Output logs to a timestamped `analysis_output_<timestamp>.txt` file.
   - Save a network visualization as `copurchase_graph.png`.
   - Save a data preparation summary as `data_prep_summary_<timestamp>.txt`.
   - Save product recommendations to a timestamped `product_recommendations_<timestamp>.txt` file (if implemented in your script).

## Output Files

- `analysis_output_<timestamp>.txt`: Full log of the analysis.
- `copurchase_graph.png`: Visualization of the product network.
- `data_prep_summary_<timestamp>.txt`: Summary of data preparation steps.
- `product_recommendations_<timestamp>.txt`: Top recommendations for influential products (if generated).

## Notes

- Make sure your data files (`copurchase.csv`, `products.csv`) match the expected format (see code for details).
- For large datasets, the script may take several minutes to complete.
- All output files will be saved in this directory.

---