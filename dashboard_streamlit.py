## TO DO
## Fix lines of best fit
## Add explanation



import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import interpolate


st.set_page_config(
     page_title="Larval Rearing Unit model",
     page_icon="🐛",
     layout="centered",
     initial_sidebar_state="expanded")


rows = 28
n_cells = rows**2

### SIDEBAR ###
st.sidebar.subheader("Settings")
eggs_per_cell = st.sidebar.slider('Intended eggs per cell', 1, 20, 10)
dispense_type = st.sidebar.selectbox('Dispense by', ('Volume', 'Number'), 
                                     help="'Volume' corresponds to a typical aliquoter in which eggs are randomly arranged in a reservoir from which aliquots of a fixed volume are taken, resulting in a variable number of eggs being dispensed into each cell (a Poisson process).\n'Number' aliquots a fixed number of eggs into each cell, the intended number of eggs into each cell.")
if dispense_type == 'Volume':
    accuracy = 0.01/3 * st.sidebar.slider('Accuracy of droplet volume', 0, 30, 15, 
                          help="The amount by which the average droplet volume dispensed into an LRU differs from the intended volume. Expressed as half of the range containing 99.7% of all average droplet volumes.")
    precision = 0.01/3 * st.sidebar.slider('Precision of droplet volume', 0, 30, 15, 
                          help="The variation between droplet volumes within a single LRU. Expressed as half of the range containing 99.7% of all droplet volumes.")
    #sensitivity = None
else:
    sensitivity = 0.025 * (10 - st.sidebar.slider('Sensitivity of dispenser', 1, 10, 5, 
                          help="How well the dispenser can sense eggs that are bunched together or are otherwise irregular"))
    #accuracy = precision = None


P_hatch = 0.01 * st.sidebar.slider('Hatch rate', 1, 100, 50, format="%d %%")
P_male = 0.01 * st.sidebar.slider('Male proportion', 1, 100, 50, format="%d %%")
dox = st.sidebar.checkbox('Doxycycline', value=False)
behaviour = st.sidebar.selectbox('Larval behaviour', ('Cannibalistic', 'Aggressive'), 
                                     help="'Cannibalistic' results in a maximum of one larva surviving per cell; 'Aggressive' results in only larvae that are alone in their cells surviving.")
P_pupate = 0.01 * st.sidebar.slider('Pupation rate', 1, 100, 90, format="%d %%")
P_collect = 0.01 * st.sidebar.slider('Pupal collection rate', 1, 100, 95, format="%d %%",
                                      help="The proportion of pupae that are successfully collected.")
fixed_rs = st.sidebar.checkbox('Fixed random seed', value=True, help="If selected, the results of random number generations will always be the same, allowing better comparison of the effects of various settings")

                                 
                
                
                
### FUNCTIONS ###

def infest(eggs_per_cell):             
    
    """Calculates eggs dispensed into cells of a Waffle, surviving males, and 
    intermediate steps
    """

    
    if(dispense_type == "Volume"):
        relative_droplet_sizes = np.random.normal(np.random.normal(1, accuracy), precision, n_cells)   
        eggs_dispensed = np.random.poisson(eggs_per_cell * relative_droplet_sizes)  # calculate number of masses in each dispense. This can be accurately modelled as a Poisson distribution rather than coding the full sampling process
    elif(dispense_type == "Number"):
        eggs_dispensed = np.zeros(n_cells)
        for i in range(eggs_per_cell):
            eggs_dispensed += 1 + np.random.poisson(sensitivity, n_cells)

    larvae1 = np.random.binomial(eggs_dispensed.astype(int), P_hatch)  # hatch eggs 
    
    if(dox):
        larvae2 = larvae1
    else:
        larvae2 = np.random.binomial(larvae1.astype(int), P_male)  # females die

       
    if(behaviour=="Cannibalistic"):
        larvae3 = larvae2 > 0  # cannibalize
    elif(behaviour=="Aggressive"):
        larvae3 = larvae2 == 1  # only solitary larvae survive

   
    pupae = np.random.binomial(larvae3, P_pupate * P_collect)
    
    utilization = np.sum(pupae)/n_cells * 100
    efficiency = utilization / eggs_per_cell
    
    results = {
      "utilization": utilization,
      "efficiency": efficiency,
      "eggs_dispensed": eggs_dispensed,
      "larvae1": larvae1,
      "larvae2": larvae2,
      "larvae3": larvae3,
      "pupae": pupae
    }

    return results

    
def plot_LRU(results, stage):
    metric = 0
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    stage = stage.replace(" ", "_")   # An issue where spaces passed from select box are not compared properly, so replace them
    if(stage == "Eggs"):
        metric = results["eggs_dispensed"]
        word = "eggs"
    elif(stage == "Larvae_(after_hatching)"):
        metric = results["larvae1"]
        word = "larvae (after hatching)"
    elif(stage == "Larvae_(after_female_die_off)"):
        metric = results["larvae2"]
        word = "larvae (after female die off)"
    elif(stage == "Larvae_(prior_to_pupation)"):
        metric = results["larvae3"]
        word = "larvae (prior to pupation)"
    elif(stage == "Pupae"):
        metric = results["pupae"]
        word = "Pupae"

                
    
    bins = np.arange(0, metric.max() + 1.5) - 0.5 
    axs[1].hist(metric, bins, edgecolor='k', linewidth=0.5)
    axs[1].set_xlabel("Number of " + word)
    axs[1].set_ylabel("Number of cells containing given number of " + word)
    axs[1].xaxis.get_major_locator().set_params(integer=True)
    axs[1].grid(axis = 'y', alpha = 0.2)  # Show grid lines
    
    metric = metric.reshape(rows, rows)
    axs[0].imshow(metric, cmap="Blues")
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    #axs[0].set_title("Example LRU showing the number of " + word + " per cell\n")

    for i in range(rows):
        for j in range(rows):
            if metric[i, j]:
                axs[0].text(j, i, metric[i, j].astype(int),
                           ha="center", va="center", color="grey", fontsize = 6)
            
    return fig


def sampling(n = 300):      
    densities = [i for i in range(1, 17+1, 1) for _ in range(50)]   # First range gives numbers, second gives repeats
    #densities = np.linspace(2, 15, n)
    utilization = np.zeros(len(densities))
    for i, density in enumerate(densities):
        results = infest(density)
        utilization[i] = results["utilization"]
    
    ## PLOT ###
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    ax1.set_xlabel('Intended eggs per cell')
    ax1.set_xticks(np.arange(1, 17.1, 1))
    
    ax1.set_ylabel('Pupal yield')
    ax1.scatter(densities, utilization, alpha=0.15)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_ylim([0,100])       
    ax1.set_yticks(np.arange(0, 100.1, 10))
    
    spl = interpolate.splrep(densities, utilization, s=20000, k=3)
    ys = interpolate.splev(densities, spl, der=0)
    ax1.plot(densities, ys, linestyle='-', linewidth=2)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(alpha = 0.2)  # Show grid lines

            
    return fig    


### MAIN PANEL ###

if(fixed_rs): np.random.seed(1)


st.title('Larval Rearing Unit model')
with st.expander("How does the model work?"):
     st.write("""
         TBD.
     """)

st.subheader('Example LRU')
st.markdown('Shows an example LRU and the distribution of insects per cell within it, based on the selected settings and the life stage selected below')
stage = st.select_slider('',
     options=['Eggs', 
              'Larvae (after hatching)', 
              'Larvae (after female die off)',
              'Larvae (prior to pupation)',
              'Pupae'])

results = infest(eggs_per_cell)
st.pyplot(plot_LRU(results, stage))



st.subheader('Performance compared to eggs aliquoted')
st.write('Shows how varying the intended number of eggs per cell affects performance')
st.pyplot(sampling())
