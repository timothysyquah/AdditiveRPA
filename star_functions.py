import yaml


# Load a YAML file
with open('setup.yaml', 'r') as file:
    setup = yaml.safe_load(file)

import sys
sys.path.append(setup['rpa_library']['path']) 
from rpa.arch import * #includes architectures to build graphs
from rpa.form import * #useful helper functions to assemble matrices
from rpa.plot import * #plotting functions
from rpa.interact import * #easy function to define interactions
from rpa.chain_tracker import * #Class that is used to account for different species and different chains. This class allows scalable RPA calculations
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans


def create_copolymer_star(NA,NB,arms=1):
    """
    Returns
    -------
    G : networkx.Graph
    """
    G = nx.Graph()
    b1 = 0
    for j in range(arms):
        for i in range(NA):
            if i==1:
                b1 = 1
                G.add_edge((NA+NB-1)*j+i, 0, b1=b1, b2=b1)
                G.add_node((NA+NB-1)*j+i, species="A",charge = 0)
            if i >1:
                b2 = 1
                G.add_edge((NA+NB-1)*j+i, (NA+NB-1)*j+(i-1), b1=b1, b2=b2)
                G.add_node((NA+NB-1)*j+i, species="A",charge = 0)

                b1 = b2
            if i==0:
                b1 = 0
                G.add_node(0, species="A",charge = 0)
        for k in range(NB):
            b1 = 1
            b2 = 1
            G.add_edge((NA+NB-1)*j+NA+k, (NA+NB-1)*j+NA+(k-1), b1=b1, b2=b2)
            G.add_node((NA+NB-1)*j+NA+k, species="B",charge = 0)
    return G


def create_miktoarm_copolymer(NA,NB,NC,arms=1):
    """
    Returns
    -------
    G : networkx.Graph
    """
    G = nx.Graph()
    b1 = 0
    for j in range(arms):
        for i in range(NA):
            if i==1:
                b1 = 1
                G.add_edge((NA+NB-1)*j+i, 0, b1=b1, b2=b1)
                G.add_node((NA+NB-1)*j+i, species="A",charge = 0)
            if i >1:
                b2 = 1
                G.add_edge((NA+NB-1)*j+i, (NA+NB-1)*j+(i-1), b1=b1, b2=b2)
                G.add_node((NA+NB-1)*j+i, species="A",charge = 0)

                b1 = b2
            if i==0:
                b1 = 0
                G.add_node(0, species="A",charge = 0)
        for k in range(NB):
            b1 = 1
            b2 = 1
            G.add_edge((NA+NB-1)*j+NA+k, (NA+NB-1)*j+NA+(k-1), b1=b1, b2=b2)
            G.add_node((NA+NB-1)*j+NA+k, species="B",charge = 0)
    N_start = (NA+NB-1)*arms
    for j in range(arms):
        for i in range(NC):
            if i==1:
                b1 = 1
                G.add_edge(N_start + (NC-1)*j+i, 0, b1=b1, b2=b1)
                G.add_node(N_start + (NC-1)*j+i, species="A",charge = 0)
            if i >1:
                b2 = 1
                G.add_edge(N_start + (NC-1)*j+i, N_start + (NC-1)*j+(i-1), b1=b1, b2=b2)
                G.add_node(N_start + (NC-1)*j+i, species="A",charge = 0)

                b1 = b2
            if i==0:
                b1 = 0
                G.add_node(0, species="A",charge = 0)
    return G


def create_copolymer_aggregate(NA,NB,n_chains=1,n_copper=1):
    """
    Returns
    -------
    G : networkx.Graph
    """
    G = nx.Graph()
    b1 = 0
    A_indices = []
    B_indices = []
    for j in range(n_chains):
        for i in range(NA):
            b1 = 1
            b2 = 1
            if i > 0:
                G.add_edge((NA+NB)*j+i, (NA+NB)*j+(i-1), b1=b1, b2=b2)
            G.add_node((NA+NB)*j+i, species="A",charge = 0)
            A_indices.append((NA+NB)*j+i)
        for k in range(NB):
            b1 = 1
            b2 = 1
            G.add_edge((NA+NB)*j+NA+k, (NA+NB)*j+NA+(k-1), b1=b1, b2=b2)
            G.add_node((NA+NB)*j+NA+k, species="B",charge = 0)
            B_indices.append((NA+NB)*j+NA+k)
    #print(A_indices)
    #print(B_indices)
    for i in range(n_copper):
        b1=1
        b2=1
        bind_indices = random.sample(range(len(A_indices)), 2)
        G.add_edge(A_indices[bind_indices[0]], A_indices[bind_indices[1]], b1=b1, b2=b2)
        #print('connecting:', A_indices[bind_indices[0]], 'and', A_indices[bind_indices[1]])
        
    if nx.number_connected_components(G) == 1:
        print('There is one polymer species')
    else:
        raise Exception("Polymer did not form a single aggregate, try re-running or increasing n_copper")
    return G

# class that is similar to Paul's function
class AdditiveWorkFlow():
    def __init__(self,chiAB, chiAS, polymer_type,Nlist,arms = 5, n_copper = 5):
        # intitializes system and computes the form factors (only do this once for a given system)
        species_to_index = {'A':0,'B':1,'S':2}
        charge_to_index = {0:0}
        
        N = sum(Nlist)
        print('N =', N)
        specieslist = ['A','B']
        chargelist = [0, 0]
        Glist = []
        Glist.append(create_salt_solvent(0,'S'))
        if polymer_type == 'linear':
            Glist.append(create_block_copolymer(Nlist,specieslist,chargelist))
        elif polymer_type == 'star':
            Glist.append(create_copolymer_star(Nlist[0],Nlist[1],arms))
        elif polymer_type == 'miktoarm':
            Glist.append(create_miktoarm_copolymer(Nlist[0],Nlist[1],Nlist[2], arms))
        elif polymer_type == 'aggregate':
            Glist.append(create_copolymer_aggregate(Nlist[0],Nlist[1],arms,n_copper))
        self.Glist = Glist
        self.species_to_index = species_to_index
        self.charge_to_index = charge_to_index
        phi_list = [1/len(Glist)]*len(Glist)
        blist = [1.0, 1.0, 1.0]
        self.ChainTracker = TrackChains(Glist,phi_list,species_to_index,charge_to_index)
        self.bLoops = self.ChainTracker.check_loops()
        if not self.bLoops:
            raise Exception("Loops are not supported")
        self.karray = np.linspace(0.001,5,100)
        bondtype = compute_bond_transition_dgc
        phi_matrix = create_bond_transition(self.karray, blist, bondtype, self.ChainTracker)
        self.Flist = []
        for i in range(len(Glist)):
            self.Flist.append(universal_compute_form_factor(Glist[i],self.karray,phi_matrix,self.ChainTracker))
        
        
        # initialize interactions
        self.zeta = 1e5
        self.a_i = 0.0
        self.Nref = 1
        self.chiAB = chiAB
        self.chiAS = chiAS
        self.phase_points = []
    # Draws the system
    def DrawSystem(self):
        fig = DrawSystem(self.Glist, self.species_to_index,'species')
        return fig

    # Update chiAb or chias
    def set_chi(self,chiAB,chiAS):
        self.chiAB = chiAB
        self.chiAS = chiAS

    # run a single point
    def run_single(self,chiBS,phiS):
        phiS = [phiS,1-phiS]
        self.ChainTracker.update_phi(phiS)
        Ftemp = self.ChainTracker.downsizeFlist(self.Flist)
        phitemp = self.ChainTracker.return_list(phiS)
        F = Assemble_F(phitemp,Ftemp,self.ChainTracker.getNlist(),self.Nref)
        interactions = []
        interactions.append({'type':'chi',0:'A',1:'B','value':self.chiAB,'smear_length':self.a_i})
        interactions.append({'type':'chi',0:'A',1:'S','value':self.chiAS,'smear_length':self.a_i})
        interactions.append({'type':'chi',0:'B',1:'S','value':chiBS,'smear_length':self.a_i})
        interactions.append({'type':'zeta','value':self.zeta,'smear_length':self.a_i})
        U = return_interaction_matrix(self.ChainTracker,interactions,self.karray)*self.Nref
        Sinv = np.linalg.inv(F)+U
        self.ChainTracker.restore()
        phase = checkphase(np.linalg.det(Sinv))
        return phase
    #run a list of points
    def run(self, phi_list, chiBS_list ):
        self.phase_points = []
        for i, chiBS in enumerate(chiBS_list):
            for j, phiS in enumerate(phi_list):
                phase = self.run_single(chiBS,phiS)
                self.phase_points.append([phiS,chiBS,phase])
        self.phase_points = np.vstack(self.phase_points)
        return self.phase_points
    # plot the phase diagram
    def plot_phase_diagram(self,fontsize = 20):
        plt.figure(figsize = (6,6))
        label = ['DIS','Macro','Micro']
        for i in range(0,3):
            loc = np.where(self.phase_points[:,-1] == i)[0]
            if len(loc)==0:
                continue
            plt.scatter(1 - self.phase_points[:,0][loc],self.phase_points[:,1][loc],label =label[i])
        plt.legend(fontsize = fontsize,bbox_to_anchor=[0.5, 0.5])
        plt.ylabel(r'$\chi_{BS}$',fontsize = fontsize)
        plt.xlabel(r'$\phi_P$',fontsize = fontsize)
        plt.xticks(fontsize = fontsize-1)
        plt.yticks(fontsize = fontsize-1)
    #export data
    def export_data(self,filename):
        np.savetxt(filename,self.phase_points,delimiter=',')
        print('Data has been saved to', filename)

def find_phase_boundaries(compositions,b_allboundary = True,class_1 = 0,class_2 = 1):
    # Extract compositions and classifications
    compositions_xy = compositions[:, :-1]
    classifications = compositions[:, -1]

    # Perform Delaunay triangulation
    tri = Delaunay(compositions_xy)
    class_3 = 1
    if class_1==0 and class_2==1:
        class_3 = 2
    elif class_1 ==1 and class_2==0:
        class_3 = 2
    elif class_1==1 and class_2==2:
        class_3 = 0
    elif class_1==1 and class_2==2:
        class_3 = 0
    # Initialize a set to store phase boundaries
    phase_boundaries = set()
    midpoints = []
    # Iterate through the triangles
    for simplex in tri.simplices:
        # Get the vertices of the triangle
        v0, v1, v2 = simplex

        # Get the classifications of the vertices
        class_v0 = classifications[v0]
        class_v1 = classifications[v1]
        class_v2 = classifications[v2]

        # Check if vertices belong to different phases
        # Check if edge 1 is a phase boundary
        if b_allboundary:

            if class_v0 != class_v1:
                # Mark edge 1 as a phase boundary
                edge1 = tuple(sorted((v0, v1)))
                phase_boundaries.add(edge1)
                # Calculate and store midpoint of edge 1
                midpoints.append((compositions_xy[edge1[0]] + compositions_xy[edge1[1]]) / 2)

            # Check if edge 2 is a phase boundary
            if class_v1 != class_v2:
                # Mark edge 2 as a phase boundary
                edge2 = tuple(sorted((v1, v2)))
                phase_boundaries.add(edge2)
                # Calculate and store midpoint of edge 2
                midpoints.append((compositions_xy[edge2[0]] + compositions_xy[edge2[1]]) / 2)

            # Check if edge 3 is a phase boundary
            if class_v2 != class_v0:
                # Mark edge 3 as a phase boundary
                edge3 = tuple(sorted((v2, v0)))
                phase_boundaries.add(edge3)
                # Calculate and store midpoint of edge 3
                midpoints.append((compositions_xy[edge3[0]] + compositions_xy[edge3[1]]) / 2)

        else:
            if class_v0 != class_v1 and class_v0 != class_3 and class_v1 != class_3:
                # Mark edge 1 as a phase boundary
                edge1 = tuple(sorted((v0, v1)))
                phase_boundaries.add(edge1)
                # Calculate and store midpoint of edge 1
                midpoints.append((compositions_xy[edge1[0]] + compositions_xy[edge1[1]]) / 2)

            # Check if edge 2 is a phase boundary
            if class_v1 != class_v2 and class_v1 != class_3 and class_v2 != class_3:
                # Mark edge 2 as a phase boundary
                edge2 = tuple(sorted((v1, v2)))
                phase_boundaries.add(edge2)
                # Calculate and store midpoint of edge 2
                midpoints.append((compositions_xy[edge2[0]] + compositions_xy[edge2[1]]) / 2)

            # Check if edge 3 is a phase boundary
            if class_v2 != class_v0 and class_v2 != class_3 and class_v0 != class_3:
                # Mark edge 3 as a phase boundary
                edge3 = tuple(sorted((v2, v0)))
                phase_boundaries.add(edge3)
                # Calculate and store midpoint of edge 3
                midpoints.append((compositions_xy[edge3[0]] + compositions_xy[edge3[1]]) / 2)

    midpoints = np.vstack(midpoints)
    return phase_boundaries, midpoints



def reduce_points(compositions, n_clusters):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(compositions)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_centers

