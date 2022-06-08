import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm


st.set_page_config(
     page_title="Larval Rearing Unit model",
     page_icon="🐛",
     layout="centered",
     initial_sidebar_state="expanded")



### SIDEBAR ###
with st.sidebar:
    st.title("Settings")
    st.subheader("Aliquoting")
    n_cells = st.select_slider('Number of cells in grid',
         options=np.square(np.arange(14, 28+1, 2)), value=784)
    rows = np.sqrt(n_cells).astype(int)
    eggs_per_cell = st.slider('Intended eggs per cell', 1, 20, 10)
    dispense_type = st.selectbox('Dispense by', ('Volume', 'Number'), 
                                         help="'Volume' corresponds to a typical aliquoter in which eggs are randomly arranged in a reservoir from which aliquots of a fixed volume are taken, resulting in a variable number of eggs being dispensed into each cell (a Poisson process).\n'Number' aliquots by counting eggs.")
    if dispense_type == 'Volume':
        accuracy = 0.01/2 * st.slider('Accuracy of droplet volume', 0, 50, 15, format="±%d %%", 
                              help="The amount by which the average droplet volume dispensed into an LRU differs from the intended volume. Expressed as 2 x the relative standard deviation, equal to the range containing 95% of all average droplet volumes.")
        precision = 0.01/2 * st.slider('Precision of droplet volume', 0, 50, 15, format="±%d %%", 
                              help="The variation between droplet volumes within a single LRU. Expressed as 2 x the relative standard deviation, equal to the range containing 95% of all droplet volumes")
    else:
        sensitivity = 0.025 * (10 - st.slider('Sensitivity of dispenser', 1, 10, 5, 
                              help="How well the dispenser can sense eggs that are bunched together or are otherwise irregular"))
    
    st.subheader("Hatching & transgene expression")
    P_hatch = 0.01 * st.slider('Hatch rate', 1, 100, 50, format="%d %%")
    P_male = 0.01 * st.slider('Male proportion', 1, 100, 50, format="%d %%")
    dox = st.checkbox('Doxycycline', value=False)
    
    st.subheader("Larval interaction")
    behaviour = st.selectbox('Larval behaviour', ('Cannibalistic', 'Aggressive'), 
                                         help="'Cannibalistic' results in a maximum of one larva surviving per cell; 'Aggressive' results in larvae that are not alone in their cell having a lower probability of survival.")
    if behaviour == 'Aggressive':
        f_aggression = st.slider('Aggression factor', 1, 4, 2,
                              help="How much the presence of additional larvae in a cell reduce the probability of a larva's survival. The survival probability is 1/N^f, where N is the number of larvae in the cell and f is the aggression factor. See below for specfic probabilities.")
        
        N = np.arange(1, 5+1)
        P = 100/np.power(N, f_aggression)
                
        with st.expander("Survival probabilities"):
             fig, ax = plt.subplots()
             ax.bar(N, P)
             ax.set_xlabel('Larvae in cell')
             ax.set_xticks(np.arange(1, 5.1, 1))
             ax.set_ylabel('Survival probability')
             ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
             ax.set_ylim([0,105])       
             ax.set_yticks(np.arange(0, 100.1, 10))
             fig.patch.set_facecolor('#F0F2F6')
             st.pyplot(fig)
    
    st.subheader("Pupation")
    P_pupate = 0.01 * st.slider('Pupation rate', 1, 100, 90, format="%d %%")
    P_collect = 0.01 * st.slider('Pupal collection rate', 1, 100, 95, format="%d %%",
                                          help="The proportion of pupae that are successfully collected.")
    
    st.subheader("Other")
    fixed_rs = st.checkbox('Fixed random seed', value=True, help="If selected, the results of random number generations will always be the same, allowing better comparison of the effects of various settings")

                                 
                
                
                
### FUNCTIONS ###

def infest(eggs_per_cell):             
    
    """Calculates eggs dispensed into cells of a Waffle, surviving males, and 
    intermediate steps
    """

    
    if(dispense_type == "Volume"):
        relative_droplet_sizes = np.random.normal(np.random.normal(1, accuracy), precision, n_cells).clip(min = 0)
        eggs = np.random.poisson(eggs_per_cell * relative_droplet_sizes)  # calculate number of masses in each dispense. This can be accurately modelled as a Poisson distribution rather than coding the full sampling process
    elif(dispense_type == "Number"):
        eggs = np.zeros(n_cells)
        for i in range(eggs_per_cell):
            eggs += 1 + np.random.poisson(sensitivity, n_cells)

    larvae1 = np.random.binomial(eggs.astype(int), P_hatch)  # hatch eggs 
    
    if(dox):
        larvae2 = larvae1
    else:
        larvae2 = np.random.binomial(larvae1.astype(int), P_male)  # females die

       
    if(behaviour=="Cannibalistic"):
        larvae3 = larvae2 > 0  # cannibalize
    elif(behaviour=="Aggressive"):
        larvae3 = np.random.binomial(larvae2.astype(int), 1/np.power(larvae2.astype(int), f_aggression).clip(min = 1))  # Addition prevents divide by0 error if no larvae in cell

   
    pupae = np.random.binomial(larvae3, P_pupate * P_collect)
    
    utilization = np.sum(pupae)/n_cells * 100
    
    results = {
      "utilization": utilization,
      "eggs": eggs,
      "larvae1": larvae1,
      "larvae2": larvae2,
      "larvae3": larvae3,
      "pupae": pupae
    }

    return results

    
def plot_LRU(results, stage):
    metric = 0
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    
    stage = stage.replace(" ", "_")   # An issue where spaces passed from select box are not compared properly, so replace them
    if(stage == "Eggs"):
        metric = results["eggs"]
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
    


def plot_performance():      
    repeats = 50
    top = 17  
    densities = np.repeat(np.arange(1, top+1), repeats)
    utilizations = np.zeros(np.shape(densities))
    eggs = np.zeros(np.shape(densities))
    for i, density in enumerate(densities):
        results = infest(density)
        utilizations[i] = results["utilization"]
        eggs[i] = np.mean(results["eggs"])
    
    ## PLOT ###
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    ax1.set_xlabel('Actual eggs per cell')
    ax1.set_xticks(np.arange(1, 100, 1))  # Can go to 100, only the required ticks will be used. Essentially sets tick spacing at 1
    ax1.set_xlim([0, 20.9])
    
    ax1.set_ylabel('Pupal yield (total pupae/number of cells)')
    ax1.scatter(eggs, utilizations, alpha=0.1)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_ylim([0,100])       
    ax1.set_yticks(np.arange(0, 100.1, 10))
    
    #LOWESS line of best fit
    l = sm.nonparametric.lowess(exog=eggs, endog=utilizations, frac=0.2)
    ax1.plot(l[:, 0], l[:, 1], linestyle='-', linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Pupal yield (total pupae)')
    ax2.set_ylim([0,n_cells])
    ax2.yaxis.grid(False)
    
    ax1.yaxis.grid(alpha = 0.3)
    ax1.xaxis.grid(alpha = 0.3)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

            
    return fig 


### MAIN PANEL ###

if(fixed_rs): np.random.seed(1)


st.title('Larval Rearing Unit model')
#with st.expander("How does the model work?"):
#     st.write("""
#         TBD.
#     """)

st.subheader('Single LRU')
st.markdown('Shows a typical larval rearing unit and the distribution of insects per cell within it, based on the selected settings and the life stage selected below')
stage = st.select_slider('',
     options=['Eggs', 
              'Larvae (after hatching)', 
              'Larvae (after female die off)',
              'Larvae (prior to pupation)',
              'Pupae'])

results = infest(eggs_per_cell)
st.pyplot(plot_LRU(results, stage))



st.subheader('Many LRUs')
st.write('Runs the model hundreds of times with different numbers of eggs per cell, with all other settings as selected, in order to show the effect on pupal yield')
st.pyplot(plot_performance())
