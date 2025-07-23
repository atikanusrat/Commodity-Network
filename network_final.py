"""
U.S. State Commodity Network Analysis
Streamlit app for visualizing state commodity similarities using Census data.
"""

import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import math  # For math.isfinite check
from config import CENSUS_API_KEY

class CommodityNetwork:
    """Analyzes and visualizes commodity flow similarities between states"""
    
    def __init__(self, year):
        """Initialize with empty graph and state info"""
        self.G = nx.Graph()  # network graph
        self.df = None       # Will hold data
        self.year = year     # Analysis year
        
        # Mapping of state names to abbreviations
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
        """Load and clean data from Census API"""
        try:
            # 1. Setup API request
            api_url = f"https://api.census.gov/data/{self.year}/cfsarea"
            params = {
                'get': 'NAME,GEO_ID,COMM,COMM_LABEL,YEAR,VAL,TON',
                'for': 'state:*',
                'key': CENSUS_API_KEY
            }
            
            # 2. Make the request
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Crash if request fails
            
            # 3. Convert to DataFrame and clean
            data = response.json()
            self.df = pd.DataFrame(data[1:], columns=data[0])
            
            # Clean the data
            self._clean_data()
            return True
            
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            return False

    def _clean_data(self):
        """Clean raw data: convert numbers, extract states, remove bad rows"""
        # Convert to numeric types
        self.df['VAL'] = pd.to_numeric(self.df['VAL'], errors='coerce')
        self.df['TON'] = pd.to_numeric(self.df['TON'], errors='coerce')
        
        # Extract state names from "Alabama - 01" format
        self.df['STATE'] = self.df['NAME'].str.split(' - ').str[0]
        self.df['STATE_ABBR'] = self.df['STATE'].map(self.state_abbreviations)
        
        # Remove summary rows and zero values
        self.df = self.df[~self.df['COMM_LABEL'].str.contains('All Commodities', na=False)]
        self.df = self.df[self.df['VAL'] > 0]

    def build_network(self):
        """Build the complete similarity network"""
        if self.df is None:
            st.error("No data loaded")
            return
            
        # 1. Create empty graph
        self.G = nx.Graph()
        
        # 2. Create state profiles (dictionary of commodities)
        state_profiles = self._create_state_profiles()
        
        # 3. Add nodes and edges
        self._add_nodes(state_profiles)
        self._connect_similar_states(state_profiles)
        
        # 4. Calculate network metrics
        self._calculate_network_stats()

    def _create_state_profiles(self):
        """Create {commodity: value} dictionaries for each state"""
        profiles = {}
        
        for state, group in self.df.groupby('STATE'):
            # Clean commodity names by removing codes like (CFS123)
            clean_commodities = group['COMM_LABEL'].str.replace(r'\(CFS\d+\)', '', regex=True)
            commodity_values = group['VAL'].values
            
            # Create dictionary of {commodity: value} pairs
            profiles[state] = dict(zip(clean_commodities, commodity_values))
            
        return profiles

    def _add_nodes(self, state_profiles):
        """Add state nodes to the network with economic metrics"""
        for state, profile in state_profiles.items():
            total_value = sum(profile.values())
            commodity_count = len(profile)
            
            # Store both value and count for sizing
            self.G.add_node(
                state,
                abbreviation=self.state_abbreviations.get(state, ''),
                total_value=total_value,
                commodity_count=commodity_count
            )

    def _calculate_similarity(self, profile1, profile2, top_n=10):
        """
        Calculate hybrid similarity score (50% top commodities, 50% all commodities)
        Returns value between 0 (no similarity) and 1 (identical)
        """
        # Get top N commodities by value for each state
        top1 = {k for k, _ in sorted(profile1.items(), key=lambda x: -x[1])[:top_n]}
        top2 = {k for k, _ in sorted(profile2.items(), key=lambda x: -x[1])[:top_n]}
        
        # Calculate two similarity components:
        # 1. Similarity of top 10 commodities
        top_sim = len(top1 & top2) / top_n  # % overlap in top commodities
        
        # 2. General similarity of all commodities
        general_sim = len(set(profile1) & set(profile2)) / len(set(profile1) | set(profile2))
        
        # Return 50/50 weighted average
        return 0.5 * top_sim + 0.5 * general_sim

    def _connect_similar_states(self, state_profiles):
        """Connect states with edges based on similarity"""
        states = list(state_profiles.keys())
        
        for i, state1 in enumerate(states):
            for state2 in states[i+1:]:
                # Calculate similarity score
                similarity = self._calculate_similarity(
                    state_profiles[state1],
                    state_profiles[state2]
                )
                
                # Only create edges for significant similarities
                if similarity > 0.7:
                    # Validate the similarity score
                    if not isinstance(similarity, (int, float)) or not math.isfinite(similarity):
                        similarity = 0.2  # Use threshold as default if invalid
                    
                    # Get top 10 commodities for tooltips
                    top1 = {k for k, _ in sorted(state_profiles[state1].items(), key=lambda x: -x[1])[:10]}
                    top2 = {k for k, _ in sorted(state_profiles[state2].items(), key=lambda x: -x[1])[:10]}
                    
                    # Add edge with metadata
                    self.G.add_edge(
                        state1, state2,
                        weight=float(similarity),  # Ensure float type
                        top_shared=len(top1 & top2),
                        general_shared=len(set(state_profiles[state1]) & set(state_profiles[state2]))
                    )
    
    def _calculate_network_stats(self):
        """Calculate metrics for visualization"""
        # 1. Node values and counts
        self.node_values = {n: d['total_value'] for n, d in self.G.nodes(data=True)}
        self.node_counts = {n: d['commodity_count'] for n, d in self.G.nodes(data=True)}
        
        # 2. Normalize for node sizing (50% value, 50% count)
        max_value = max(self.node_values.values()) if self.node_values else 1
        max_count = max(self.node_counts.values()) if self.node_counts else 1
        
        self.node_sizes = {
            n: 0.5 * (self.node_values[n]/max_value) + 0.5 * (self.node_counts[n]/max_count)
            for n in self.G.nodes()
        }
        
        # 3. Calculate centrality (importance in network)
        self.node_degrees = dict(self.G.degree(weight='weight'))
        max_degree = max(self.node_degrees.values()) if self.node_degrees else 1
        self.node_centrality = {n: d/max_degree for n, d in self.node_degrees.items()}

    def visualize_network(self):
        """Create interactive Plotly visualization"""
        if not self.G.nodes():
            return None
            
        # 1. Calculate node positions
        pos = nx.spring_layout(self.G, seed=42)  # Consistent layout
        
        # 2. Prepare edge coordinates
        edge_x, edge_y = [], []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # None creates line breaks
            edge_y.extend([y0, y1, None])
        
        # 3. Create node trace
        node_trace = go.Scatter(
            x=[pos[n][0] for n in self.G.nodes()],
            y=[pos[n][1] for n in self.G.nodes()],
            mode='markers+text',
            text=[self.state_abbreviations.get(n, '') for n in self.G.nodes()],
            textposition="middle center",
            marker=dict(
                size=[10 + 40 * self.node_sizes[n] for n in self.G.nodes()],  # Scale node size
                color=[self.node_centrality[n] for n in self.G.nodes()],  # Color by centrality
                colorscale='Viridis',
                line=dict(width=1, color='gray')
            ),
            hoverinfo='text',
            hovertext=[self._get_node_info(n) for n in self.G.nodes()]
        )
        
        # 4. Create edge trace with bounded widths
        edge_trace = go.Scatter(
            x=edge_x, 
            y=edge_y,
            line=dict(
                # Use a fixed width for all edges (Plotly does not support per-edge width in a single trace)
                width=2,
                color='rgba(150, 150, 150, 0.7)'  # Semi-transparent gray
            ),
            hoverinfo='text',
            hovertext=[
                f"{e[0]} ↔ {e[1]}<br>"
                f"Similarity: {self.G.edges[e]['weight']:.2f}<br>"
                f"Top-10 shared: {self.G.edges[e]['top_shared']}/10 | "
                f"All shared: {self.G.edges[e]['general_shared']}"
                for e in self.G.edges
            ],
            mode='lines'
        )
        
        # 5. Combine into figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="State Commodity Similarity Network<br>"
                      "Node size: 50% value + 50% commodity count | "
                      "Edge weight: 50% top-10 + 50% all commodities",
                titlefont_size=12,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            )
        )
        return fig

    def _get_node_info(self, node):
        """Generate hover text for nodes"""
        return (
            f"<b>{node}</b><br>"
            f"Abbreviation: {self.G.nodes[node]['abbreviation']}<br>"
            f"Total Value: ${self.node_values.get(node, 0):,.0f}<br>"
            f"Commodity Count: {self.node_counts.get(node, 0):,}<br>"
            f"Network Centrality: {self.node_centrality.get(node, 0):.2f}"
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
        **Node Size:** Total commodity value + commodity count  
        **Node Color:** Network centrality (importance)  
        **Edges:** Similarity between states (thicker = more similar)
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
            .nlargest(10, 'VAL')
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
