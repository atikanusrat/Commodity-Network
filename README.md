
# U.S. State Commodity Network Analysis
<img width="1426" height="715" alt="Screenshot 2025-07-23 at 6 50 49 PM" src="https://github.com/user-attachments/assets/e89874e3-c3eb-4bf4-974a-4ae6536b80e8" />

*An interactive Streamlit application that visualizes economic relationships between U.S. states based on commodity trade patterns, using Census Bureau data.*

## Key Features

- **Network Visualization**:
  - Nodes represent states (size = economic importance)
  - Edges show trade similarity (thickness = connection strength)
  - Color indicates network centrality (influence)

- **Interactive Elements**:
  - Hover over states for detailed metrics
  - Select specific states to view their top commodities
  - Compare different years (2012 vs 2017)

- **Analytics**:
  - Hybrid similarity scoring (top commodities + overall profile)
  - Economic importance calculations
  - Network centrality metrics

## How It Works

1. **Data Pipeline**:
   - Fetches commodity flow data from Census API
   - Cleans and processes trade values
   - Creates state trade profiles

2. **Network Analysis**:
   - Calculates similarity between state trade portfolios
   - Identifies strongest economic relationships
   - Computes each state's network influence

3. **Visualization**:
   - Spring-layout force-directed graph
   - Dynamic node sizing and coloring
   - Interactive tooltips and selection

## Installation

1. Get a [Census API key](https://api.census.gov/data/key_signup.html)
2. Create `config.py` with your key:
   ```python
   CENSUS_API_KEY = "your_api_key_here"
Install requirements:

bash
pip install streamlit networkx pandas plotly requests
Usage
bash
streamlit run network_final.py
Then interact with:

Year selector in sidebar

State dropdown for detailed views

Hover tooltips on nodes/edges

**Data Sources**
Primary: U.S. Census Bureau Commodity Flow Survey

Years Available: 2012, 2017

**Customization Options**
In network_final.py you can adjust:

_connect_similar_states(): Change similarity threshold (currently 0.7)

visualize_network(): Modify visual styling

_calculate_similarity(): Adjust weighting between top and general commodities
