
# U.S. State Commodity Network Analysis
<img width="1408" height="712" alt="Screenshot 2025-07-23 at 3 48 40 PM" src="https://github.com/user-attachments/assets/061a40c2-95d6-4bbe-95b4-8b5255d7e0ca" />

*Interactive visualization of state trade similarities*

## Overview
This Streamlit app analyzes and visualizes interstate commodity flow patterns using U.S. Census data. It identifies states with similar trade profiles and calculates their network centrality.

## Features
- Interactive network visualization (node size = trade volume, color = centrality)
- State-specific trade metrics (total value, top commodities)
- Similarity-based connections between states

## Project Structure
.
├── README.md # Project documentation (you are here)
├── config.py # Configuration (API key)
├── network_final.py # Main application code

text

## Requirements
**None** (No special hardware/OS requirements)

## Required Packages
streamlit==1.32.0
networkx==3.2.1
pandas==2.1.0
plotly==5.18.0
requests==2.31.0
numpy==1.26.0

text

## Setup & Usage
1. **Add your Census API key**:
   - Edit `config.py` and replace `YOUR_API_KEY_HERE` with your key from [Census API Signup](https://api.census.gov/data/key_signup.html).

2. **Install packages**:
   ```bash
   pip install -r requirements.txt
Run the app:

bash
streamlit run network_final.py
Interact with the app:

Select a year (2017 or 2012) in the sidebar

Hover over nodes to see state details

Click the "Select State" dropdown to view specific trade data

# Data Source
U.S. Census Bureau Commodity Flow Survey (CFS) via public API.
