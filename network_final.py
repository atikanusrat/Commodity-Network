"""
U.S. State Commodity Network Analysis
Streamlit app for visualizing interstate commodity flow patterns using Census data.
"""

import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
from config import CENSUS_API_KEY  

class CommodityNetwork:
    """
    Simplified network analysis of state commodity flows.
    
    Attributes:
        G (nx.Graph): Network graph
        df (pd.DataFrame): Commodity flow data
        year (int): Analysis year
        state_abbreviations (dict): State name to abbreviation mapping
    """
    
    def __init__(self, year):
        """Initialize with empty graph and basic state info"""
        self.G = nx.Graph()  # Using undirected graph for simplicity
        self.df = None
        self.year = year
        
        # State abbreviations for display
        self.state_abbreviations = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY'
        }
        
    def load_data(self):
        """Load data from Census API and clean it"""
        try:
            # API request setup
            api_url = f"https://api.census.gov/data/{self.year}/cfsarea"
            params = {
                'get': 'NAME,GEO_ID,COMM,COMM_LABEL,YEAR,VAL,TON',
                'for': 'state:*',
                'key': CENSUS_API_KEY  # From config file
            }
            
            # Get data from API
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            # Convert to DataFrame
            data = response.json()
            self.df = pd.DataFrame(data[1:], columns=data[0])
            
            # Clean the data
            self._clean_data()
            return True
            
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            return False
    
    def _clean_data(self):
        """Clean and prepare the raw data"""
        # Convert numeric columns
        self.df['VAL'] = pd.to_numeric(self.df['VAL'], errors='coerce')
        self.df['TON'] = pd.to_numeric(self.df['TON'], errors='coerce')
        
        # Extract state names (format: "Alabama - 01")
        self.df['STATE'] = self.df['NAME'].str.split(' - ').str[0]
        self.df['STATE_ABBR'] = self.df['STATE'].map(self.state_abbreviations)
        
        # Remove aggregate entries and zero values
        self.df = self.df[~self.df['COMM_LABEL'].str.contains('All Commodities', na=False)]
        self.df = self.df[self.df['VAL'] > 0]
    
    def build_network(self):
        """Build the state similarity network"""
        if self.df is None:
            st.error("No data loaded")
            return
            
        self.G = nx.Graph()
        state_profiles = self._create_state_profiles()
        self._add_nodes(state_profiles)
        self._connect_similar_states(state_profiles)
        self._calculate_network_stats()
    
    def _create_state_profiles(self):
        """Create commodity profiles for each state"""
        profiles = {}
        
        for state, group in self.df.groupby('STATE'):
            # Clean commodity names by removing codes like (CFS123)
            clean_commodities = group['COMM_LABEL'].str.replace(r'\(CFS\d+\)', '', regex=True)
            commodity_values = group['VAL'].values
            
            # Create dictionary of {commodity: value} pairs
            profiles[state] = dict(zip(clean_commodities, commodity_values))
            
        return profiles
    
    def _add_nodes(self, state_profiles):
        """Add state nodes to the network"""
        for state, profile in state_profiles.items():
            total_value = sum(profile.values())
            self.G.add_node(
                state,
                abbreviation=self.state_abbreviations.get(state, ''),
                total_value=total_value
            )
    
    def _connect_similar_states(self, state_profiles):
        """Connect states with similar commodity profiles"""
        states = list(state_profiles.keys())
        
        # Compare each pair of states
        for i, state1 in enumerate(states):
            for state2 in states[i+1:]:
                similarity = self._calculate_similarity(
                    state_profiles[state1],
                    state_profiles[state2]
                )
                
                # Only add edges for meaningful similarities
                if similarity > 0.1:  # Threshold can be adjusted
                    self.G.add_edge(state1, state2, weight=similarity)
    
    def _calculate_similarity(self, profile1, profile2):
        """
        Simple similarity calculation between two state profiles.
        
        Returns a value between 0 (no similarity) and 1 (identical).
        """
        common_commodities = set(profile1.keys()) & set(profile2.keys())
        
        if not common_commodities:
            return 0
            
        # Calculate similarity based on shared commodities
        total_possible = sum(profile1.values()) + sum(profile2.values())
        shared_value = sum(min(profile1[c], profile2[c]) for c in common_commodities)
        
        return (2 * shared_value) / total_possible if total_possible > 0 else 0
    
    def _calculate_network_stats(self):
        """Calculate basic network statistics"""
        self.node_values = {n: d['total_value'] for n, d in self.G.nodes(data=True)}
        self.node_degrees = dict(self.G.degree(weight='weight'))
        
        # Simple centrality calculation (normalized degree)
        max_degree = max(self.node_degrees.values()) if self.node_degrees else 1
        self.node_centrality = {n: d/max_degree for n, d in self.node_degrees.items()}
    
    def visualize_network(self):
        """Create interactive network visualization"""
        if not self.G.nodes():
            return None
            
        # Node positions for visualization
        pos = nx.spring_layout(self.G, seed=42)  # Consistent layout
        
        # Prepare node trace
        node_trace = go.Scatter(
            x=[pos[n][0] for n in self.G.nodes()],
            y=[pos[n][1] for n in self.G.nodes()],
            mode='markers+text',
            text=[self.state_abbreviations.get(n, '') for n in self.G.nodes()],
            textposition="middle center",
            marker=dict(
                size=[15 + 30 * (self.node_values[n]/max(self.node_values.values())) 
                     for n in self.G.nodes()],
                color=[self.node_centrality[n] for n in self.G.nodes()],
                colorscale='Viridis',
                line=dict(width=1, color='gray')
            ),
            hoverinfo='text',
            hovertext=[self._get_node_info(n) for n in self.G.nodes()]
        )
        
        # Prepare edge trace
        edge_x, edge_y = [], []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
        )
        
        return fig
    
    def _get_node_info(self, node):
        """Generate hover text for a node"""
        return (
            f"<b>{node}</b><br>"
            f"Abbreviation: {self.G.nodes[node]['abbreviation']}<br>"
            f"Total Value: ${self.node_values.get(node, 0):,.0f}<br>"
            f"Centrality: {self.node_centrality.get(node, 0):.2f}<br>"
            f"Connections: {self.node_degrees.get(node, 0):.0f}"
        )

def main():
    """Main application interface"""
    st.set_page_config(page_title="Commodity Network", layout="wide")
    st.title("U.S. State Commodity Flow Network")
    
    # Year selection
    year = st.sidebar.selectbox("Select Year", [2017, 2012])
    
    # Initialize network
    if 'network' not in st.session_state or st.session_state.get('year') != year:
        st.session_state.network = CommodityNetwork(year)
        if st.session_state.network.load_data():
            st.session_state.network.build_network()
            st.session_state.year = year
        else:
            st.error("Failed to load data")
            return
    
    network = st.session_state.network
    
    # Main visualization
    st.subheader(f"{year} State Commodity Similarity Network")
    st.markdown("""
        **Node Size:** Total commodity value  
        **Node Color:** Network centrality  
        **Edges:** Similarity between state commodity profiles
    """)
    
    fig = network.visualize_network()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # State details section
    if network.G.nodes():
        selected_state = st.selectbox("Select a state to view details", sorted(network.G.nodes()))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Commodity Value", f"${network.node_values.get(selected_state, 0):,.0f}")
        with col2:
            st.metric("Network Centrality", f"{network.node_centrality.get(selected_state, 0):.2f}")
        
        # Top commodities table
        st.subheader(f"Top Commodities in {selected_state}")
        state_data = network.df[network.df['STATE'] == selected_state].copy()
        
        # Clean and aggregate data
        state_data['COMM_LABEL'] = state_data['COMM_LABEL'].str.replace(r'\(CFS\d+\)', '', regex=True)
        top_commodities = (
            state_data.groupby('COMM_LABEL')
            .agg({'VAL': 'sum', 'TON': 'sum'})
            .nlargest(5, 'VAL')
            .reset_index()
        )
        
        # Format for display
        top_commodities['VAL'] = top_commodities['VAL'].apply("${:,.0f}".format)
        top_commodities['TON'] = top_commodities['TON'].apply("{:,.0f} tons".format)
        
        st.table(top_commodities.rename(columns={
            'COMM_LABEL': 'Commodity',
            'VAL': 'Total Value',
            'TON': 'Total Tonnage'
        }))

if __name__ == "__main__":
    main()