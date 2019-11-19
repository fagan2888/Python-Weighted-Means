import numpy as np

def group_adjust(vals,groups,weights):

    #Check if inputs are appropriate
    for j in groups:
        if len(j) != len(vals):
            raise ValueError('Size of each group must be same as size of val list')
    if len(groups) != len(weights):
        raise ValueError('Number of weights must equal number of groups')

    #Make lists into arrays for computational efficiency
    vals = np.array(vals)
    groups = np.array(groups)

    #Find indices in list where value is np.NaN
    value_list = np.where(np.isfinite(vals))[0]

    group_means = []
    for i in range(len(groups)):
        #Find indices of unique elements
        _, indices = np.unique(groups[i], return_inverse=True)

        #Make array of np.NaN of size of vals
        group_avg = np.empty(len(vals))
        group_avg.fill(np.NaN)

        indices = np.array(indices)
        #For each unique group element, get mean of vals for indices where that element is present and vals is finite
        for j in np.unique(indices):
            val_indices = np.where(indices == j)[0]
            group_avg[np.intersect1d(val_indices,value_list)] = np.mean(vals[np.intersect1d(val_indices,value_list)])
        group_means.append(group_avg)

    #Array of means for each element at all its locations in its group (where vals is finite)
    group_means = np.array(group_means)

    #Demeaned values
    weights = np.array(weights)
    final_vals = np.array(vals) - weights.dot(group_means)

    return final_vals
