# Python notebooks used for analysis of connectomic data

***Synapse tables and other relevant data files are required to run these notebooks. Once downloaded, they should be put in the /connectome_analysis/data directory. These can be obtained from: https://www.dropbox.com/scl/fo/sru3wxvbdos49rs68iqpk/AIxyK7mZqQvu4YQQizpVL8c?rlkey=pu1rh5ool7ol4a3buzawgrkgb&st=mferg5at&dl=0***

The `*_syntable*.csv` files contain one presynaptic-to-postsynaptic connection
per row, including partner information and coordinates. The `*_conntable*.csv`
files aggregate those rows into unique neuron-pair connections and store the
number of synapses in the `count` column. 

- bee_fly_analysis.ipynb: Figure 2j,h-i; Figure 4c-j; Extended Data Figure 4; Extended Data Figure 6 

- EB_synapse_distributions.ipynb: Figure 2g

- electrotonic_distance.ipynb: Figure 2k
