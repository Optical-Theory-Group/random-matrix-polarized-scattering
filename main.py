import os
from functions import (get_mode_keys, get_statistics, mode_sample_cartesian, S2M, S_sampler_svd, S_cascade, M2S, mie_cross_section)
import numpy as np
import scipy as sp
import h5py

def get_data(f, data_blocks, cascade_method):
    '''
    Performs polarimetric calculations of different blocks of the scattering matrices and saves data.
    
    Input parameters:
    'f': the hdf5 file where the data is saved. Used to extract the scattering matrices
    'data_blocks': list of which blocks of the scattering matrix data will be saved for
    'cascade_method': either 'S' or 'M' depending on whether scattering or transfer matrices are being used
    '''
    
   
    matrices = f['Random Matrices']['Matrices']
    _, N, _ = np.shape(f['Random Matrices']['Single Pool M'])
    n_modes = int(N/4)
    n_times, _, _ = np.shape(matrices)
    
    data = {}
    data['Transmission'] = np.zeros((n_times), dtype=float)
    data['Tau'] = np.zeros((2*n_modes*n_times), dtype=np.complex128)
        
    for block in data_blocks:
        block_str = str(block)
        data[block_str] = {}
        data[block_str]['Diattenuation'] = np.zeros((n_times), dtype=float)
        data[block_str]['Retardance'] = np.zeros((n_times), dtype=float)
        data[block_str]['Mean'] = np.zeros((2,2), dtype=np.complex128)
        data[block_str]['H'] = np.zeros((4,4), dtype=np.complex128)
        data[block_str]['Matrices'] = np.zeros((n_times,2,2), dtype=np.complex128)
    
    for p in range(n_times):
        if cascade_method == 'M':
            S = M2S(matrices[p,:,:])
        elif cascade_method == 'S':
            S = matrices[p,:,:]
            
        # Extract subblocks
        t = S[2*n_modes: 4*n_modes, 0:2*n_modes]
        t2 = S[0:2*n_modes, 2*n_modes:4*n_modes]
        r = S[0:2*n_modes, 0:2*n_modes]
        r2 = S[2*n_modes:4*n_modes, 2*n_modes:4*n_modes]

        # Gather data for each mode
        for block in data_blocks:
            k = block[0]
            j = block[1]
            i = block[2]     
            block_str = str(block)
            
            # Pick correct matrix
            if k == 'r':
                mat = r
            elif k == 't':
                mat = t
            elif k == 't2':
                mat = t2
            elif k == 'r2':
                mat = r2
            
            # Extract 2x2 block of interest
            T = mat[2*j:2*j+2, 2*i:2*i+2]            
            data[block_str]['Matrices'][p,:,:] = T
            
            # Calculate diattenuation and retardance
            TR, TD = sp.linalg.polar(T)
        
            R_eigenvalues, R_eigenvectors = np.linalg.eig(TR)
            theta1 = np.angle(R_eigenvalues[0])
            theta2 = np.angle(R_eigenvalues[1])
            R = min(abs(theta1 - theta2), 2*np.pi - abs(theta1 - theta2 ))
            data[block_str]['Retardance'][p] = R
    
            D_eigenvalues, D_eigenvectors = np.linalg.eig(TD)
            s1 = np.abs(D_eigenvalues[0])
            s2 = np.abs(D_eigenvalues[1])
            smax = max(s1,s2)
            smin = min(s1,s2)
            D = (smax**2 - smin**2)/(smax**2 + smin**2)
            data[block_str]['Diattenuation'][p] = D

            # Calculate mean and correlation matrices
            data[block_str]['Mean'][:,:] = data[block_str]['Mean'][:,:] + T 
            v = np.ndarray.flatten(T)
            H = np.outer(v, np.conj(v))
            data[block_str]['H'][:,:] = data[block_str]['H'][:,:] + H
            
        # Mean transmission
        trans = np.real(np.trace(t@t.conj().T)/(2*n_modes))
        data['Transmission'][p] = trans
        
        tau = np.real(sp.linalg.eigvals(np.conj(t.T)@t))
        for num, x in enumerate(tau):
            data['Tau'][p*2*n_modes + num] = x

    for block in data_blocks:
        block_str = str(block)
        data[block_str]['Mean'][:,:] = data[block_str]['Mean'][:,:]/n_times
        data[block_str]['H'][:,:] = data[block_str]['H'][:,:]/n_times
       
    return data

def main():
    
    #########################
    # Simulation Parameters #
    #########################
    
    # root is the directory in which all of the data is stored (should be existing folder)
    root = r'/home/niall/simulations/data/'
        
    # name of the hdf5 file in which the data is saved
    hdf5_filename = 'data.hdf5'
    
    # wavelength, wavenumber and mean free path
    lam = 500e-9
    k = 2*np.pi/lam
    
    # normalized k-space lattice spacing (boundary is x^2 + y^2 = 1)
    # weight is the normalized integration weight (sum = pi) 
    # on_axis_index is the index of the mode [0,0,1], i.e. an on axis plane wave
    dx = 0.8
    dy = 0.8
    mode_list, weight = mode_sample_cartesian(dx, dy)
    n_modes = len(mode_list)
    on_axis_index = int((n_modes-1)/2)
    mat_size = 4*n_modes
    
    # cascade method is either 'S' or 'M' and keeps track of what kind of matrix is being used
    # 'M' = transfer matrix, 'S' = scattering matrix
    cascade_method = 'M'
    
    # n_times = number of scattering matrix realizations used in computing statistics
    # n_single_pool = number of matrices for media of thickness dl
    # n_multi_pool = number of matrices for media of thickness equal to the simulation step size
    n_times = 1*10**1
    n_single_pool = 10**1
    n_multi_pool = 10**1

    
    # matrix used for reciprocity symmetry
    # see reciprocity section of https://doi.org/10.1103/PhysRevResearch.3.013129 for more details
    sigma_p = np.zeros((4*on_axis_index+2, 4*on_axis_index+2), dtype = np.complex128)
    for i in range(4*on_axis_index+2):
        if i%2 != 0:
            sigma_p[i, 4*on_axis_index+1 - i+1] = 1
        else:
            sigma_p[i, 4*on_axis_index+1 -i-1] = 1

    # Generate 'keys' for keeping track of modes
    # example usage: mode_keys['t'][j,i] shows the outgoing and incoming wavevectors associated with the 
    # transmission matrix block t[j,i]
    mode_keys = get_mode_keys(mode_list)
        
    # Designate matrix blocks for which data will be recorded
    # first element designates the block of S ('r', 't', 't2' or 'r2')
    # the second and third elements (j,i) are the indices (t[j,i]) 
    data_blocks = [['r', on_axis_index, on_axis_index],
                   ['r', on_axis_index+1, on_axis_index],
                   ['t', on_axis_index, on_axis_index],
                   ['t', on_axis_index+1, on_axis_index]]
                          
    # Physical parameters
    # x = size parameter
    # m = relative refractive index
    # vol_frac = volume fraction of scatterers

    Mie2 = {
        'name' : 'Mie2',
        'lam' : lam,
        'k' : k,
        'x' : 2.0,
        'm' : 1.2,
        'vol_frac' : 0.01,
        'type' : 'mie',
        'pathname' : root + 'Mie2/',
        'modes': mode_list,
        'keys' : mode_keys,
        'data_blocks' : data_blocks,
        'L final' : 30.5,
        'L spacing' : 0.5,
        'tol': 0.1,
        'weight': weight,
        'n cores': 1
        }
    
    Chiral4 = {
        'name' : 'Chiral4',        
        'lam' : lam,
        'k' : k,
        'x' : 4.0,
        'vol_frac' : 0.01,
        'type' : 'chiral',
        'pathname' : root + 'Chiral4/',
        'modes': mode_list,
        'keys' : mode_keys,
        'data_blocks' : data_blocks,
        'L final' : 10.5,
        'L spacing' : 0.05,
        'mL' : 1.244,
        'mR' : 1.156,
        'm' : 1.2,
        'tol': 0.1,
        'weight': weight,
        'n cores': 1
        }

    params_array = [Mie2, Chiral4]
    
    # Loop over params_array for multiple simulation runs
    for params in params_array:

        print(f'Starting {params["name"]}')
        
        ############################################################
        # Unpack dictionaries and calculatea additional parameters #
        ############################################################
        
        pathname = params['pathname']
        data_blocks = params['data_blocks']
        lam = params['lam']
        k = params['k']
        x = params['x']
        m = params['m']
        vol_frac = params['vol_frac']
        tol = params['tol']
        scattering_type = params['type']
        
        # n is the particle volume density
        n = 3/4 * vol_frac * k**3 / (np.pi*x**3)
        params['n'] = n
        
        # theoretical cross section from Mie theory
        C_sca = mie_cross_section(x, m, k)[0]
        params['C_sca'] = C_sca
        
        # mean free path
        mfp = 1/(n*C_sca)
        params['mfp'] = mfp
        
        # particle radius
        a = x/k
        params['a'] = a
        
        # particle volume
        V = 4/3*np.pi*a**3
        params['V'] = V
        
        # mean volume occupied per particle
        Vpp = 1/params['n']
        params['Vpp'] = Vpp
        
        # mean particle spacing
        d = Vpp**(1/3)
        params['d'] = d
        name = params['name']

        #########################################################
        # Generate covariance matrix and cholesky decomposition #
        #########################################################
        
        # create folders for data storage
        if not os.path.isdir(pathname):
            os.mkdir(pathname)        

        print('Generating statistics...')
        statistics, dL = get_statistics(params)
        params['dL'] = dL
       
        with open(pathname + 'statistics.npy', 'wb') as f:
            np.save(f, statistics)

        ##################
        # Physics checks #
        ##################
        
        print('\nSanity checks:')
        # check that parameters make physics sense
        run_flag = True
        
        # 1, far feild check
        kd = k*d
        print(f'kd = {kd}')
        if kd > max(1, 0.5*x**2):
            print('Far field condition satisfied...')
        else:
            print('Particles too close. Far field condition violated')
            run_flag = False

        # 2, weak scattering regime check
        kl = k*mfp   
        print(f'kl = {kl}')
        if kl > 1.0:
            print('Weak scattering regime condition satisfied...')
        else:
            print('Weak scattering condition violated.')
            run_flag = False
            
        # 3, slab thickness phase condition check
        kL = k*dL
        print(f'kL = {kL}')        
        if kL > 1.0:
            print('Slab thick enough for z phase variation...')
        else:
            print('Insufficient thickness for phase variation.')
            run_flag = False
            
        # 4, particle radius check
        para4 = dL/(2*a)
        print(f'dL/2r = {para4}')
        if para4 > 1.0:
            print('Slab thick enough for particles to fit...')
        else:
            print('Slab too thin for particles to fit.')
            run_flag = False
            
        # 5, single scattering check
        para5 = dL/mfp
        print(f'dL/l = {para5}')
        if para5 < 1:
            print('Slab thin enough for single scattering regime...')
        else:
            print('Slab too thick, single scattering assumption violated.')
            run_flag = False 
            
        # Save record of useful parameters   
        with open(os.path.join(pathname,'params.txt'), 'w') as f:
            f.write('Input parameters:\n')
            f.write(f'Wavelength: {lam}\n')
            f.write(f'Size Parameter: {x}\n')
            f.write(f'Relative Refractive Index: {m}\n')
            f.write(f'Volume Fraction: {vol_frac}\n')            
            if params['type'] == 'chiral':
                f.write(f'mL: {params["mL"]}\n')
                f.write(f'mR: {params["mR"]}\n')
            f.write(f'Tolerance: {tol}\n')
            f.write(f'Type: {scattering_type}\n')
            f.write(f'n_times: {n_times}\n')
            f.write(f'Number of modes: {n_modes}\n')
            f.write(f'dx: {dx}\n')
            f.write(f'dy: {dy}\n')
            f.write(f'Single Pool Size: {n_single_pool}\n')
            f.write(f'Multi Pool Size: {n_multi_pool}\n')
            
            f.write('\nCalculated Parameters\n')
            f.write(f'Cross Section: {C_sca}\n')
            f.write(f'Slab Thickness: {dL}\n')
            f.write(f'Wavenumbver: {k}\n')
            f.write(f'Particle Radius: {a}\n')
            f.write(f'Particle Volume: {V}\n')
            f.write(f'Density: {n}\n')
            f.write(f'Volume Per Particle: {Vpp}\n')
            f.write(f'Particle Separation: {d}\n')
            f.write(f'Mean Free Path: {mfp}\n')
            
            f.write('\nPhysical Checks\n')
            f.write('Far Field\n')
            f.write(f'kd = {kd}\n')
            f.write('Weak Scattering\n')
            f.write(f'kl = {kl}\n')
            f.write('z Phase Variation\n')
            f.write(f'kL = {kL}\n')
            f.write('Particle Fitting\n')
            f.write(f'L/2r = {para4}\n')
            f.write('Single Scattering\n')
            f.write(f'kl = {para5}')

        if run_flag:
            print('Physical parameters all reasonable\n')
        else:
            print('One or more physical checks violated. Terminating...')
            return

        #######################################################
        # Different thicknesses visited during the simulation #
        #######################################################
        
        # L_start, L_final and L_spacing are all in units of mfp        
        
        dL_mfp = dL/mfp
        L_final = params['L final']
        L_spacing = params['L spacing']        
        mat_spacing = int(np.round(L_spacing/dL_mfp))
        n_mat_array = np.array([1] + [i*mat_spacing for i in range(1,int(np.round(L_final/L_spacing)))])
        L_array = n_mat_array*dL_mfp        
        n_steps = len(n_mat_array)
        
        # Set up propagator matrices
        # kz_list is a list of all the z components of the wavevectors for each plane wave mode
        kz_list = [mode_keys['t'][i,0][0][2] for i in range(n_modes)]
        exponential_list = np.array([np.exp(1j*k*kz*dL) for kz in kz_list])
        lambda_plus = np.kron(np.diag(exponential_list), np.identity(2))
        lambda_minus = np.conj(lambda_plus.T)
        lambda_plus_minus = np.block([[lambda_plus, np.zeros((2*n_modes, 2*n_modes))],[np.zeros((2*n_modes, 2*n_modes)), lambda_minus]])
        
        ###########################################
        # Set up hdf5 file for storing everything #
        ###########################################
        
        with h5py.File(pathname + hdf5_filename, 'w') as f:
            
            # Random matrices group saves random matrix pools and working matrices using for calculating statistics
            matrix_group = f.create_group('Random Matrices')
            _ = matrix_group.create_dataset('Single Pool M', (n_single_pool, mat_size, mat_size), dtype = np.complex128)        
            _ = matrix_group.create_dataset('Single Pool S', (n_single_pool, mat_size, mat_size), dtype = np.complex128)        
            _ = matrix_group.create_dataset('Multi Pool S', (n_multi_pool, mat_size, mat_size), dtype = np.complex128)
            _ = matrix_group.create_dataset('Multi Pool M', (n_multi_pool, mat_size, mat_size), dtype = np.complex128)        
            _ = matrix_group.create_dataset('Matrices', (n_times, mat_size, mat_size), dtype = np.complex128)        
            matrix_group['Matrix Type'] = cascade_method

            # data group saves all statistial data calculated from the 'Matrices' array within the matrix group
            # Transmission = mean sum of transmission eigenvalues
            # tau = transmission eigenvalues (different realizations)
            # More things can be added here if needed
            data_group = f.create_group('Data')
            _ = data_group.create_dataset('Thicknesses', (n_steps,), dtype=float, data=L_array)
            _ = data_group.create_dataset('n_matrices', (n_steps,), dtype=int, data=n_mat_array)
            _ = data_group.create_dataset('Transmission', (n_steps, n_times), dtype=float)
            _ = data_group.create_dataset('Tau', (n_steps, n_times*2*n_modes), dtype=np.complex128)

            for block in data_blocks:
                data_subgroup = data_group.create_group(str(block))
                data_subgroup['Mode'] = str(block)
                _ = data_subgroup.create_dataset('Diattenuation', (n_steps, n_times), dtype=float)
                _ = data_subgroup.create_dataset('Retardance', (n_steps, n_times), dtype=float)
                _ = data_subgroup.create_dataset('Mean', (n_steps, 2,2), dtype=np.complex128)
                _ = data_subgroup.create_dataset('H', (n_steps, 4,4), dtype=np.complex128)
                _ = data_subgroup.create_dataset('Matrices', (n_steps, n_times, 2,2), dtype=np.complex128)
                
        #################################
        # Make pools of random matrices #
        #################################
                
        print('Generating single pool...')
        
        with h5py.File(pathname + hdf5_filename, 'r+') as f:
            matrix_group = f['Random Matrices']
            
            for p in range(n_single_pool):
                
                # Show progress (every 10%)
                if p+1 in [int(n_single_pool*n/10) for n in range(1,11)]:
                    print(str(p+1) + '/' + str(n_single_pool))

                S_new = S_sampler_svd(statistics, sigma_p)
                M_new = S2M(S_new)
                M = lambda_plus_minus@M_new
                matrix_group['Single Pool M'][p,:,:] = M
                
            print('Single pool generation complete...\n')
            print('Generating multi pool...')
        
            for p in range(n_multi_pool): 
                
                # Show progress every 10%
                if p+1 in [int(n_multi_pool*n/10) for n in range(1,11)]:
                    print(str(p+1) + '/' + str(n_multi_pool))
    
                
                new_multi_pool_mat = np.identity(mat_size, dtype=np.complex128)
                
                # For each matrix in the multi pool we cascade mat_spacing matrices from the single pool
                for q in range(mat_spacing):
                    index = np.random.randint(0,n_single_pool)
                    M_new = matrix_group['Single Pool M'][index,:,:]
                    new_multi_pool_mat = new_multi_pool_mat@M_new
                    
                matrix_group['Multi Pool M'][p,:,:] = new_multi_pool_mat                    
                matrix_group['Multi Pool S'][p,:,:] = M2S(new_multi_pool_mat)

        print('Pool generation complete...\n')
        
        #################################################
        # Initialize matrix array and load correct pool #
        #################################################

        print('Initializing matrix arrays...')

        with h5py.File(pathname + hdf5_filename, 'r+') as f:
            matrix_group = f['Random Matrices']

            for p in range(n_times):
                if cascade_method == 'M':
                    matrix_group['Matrices'][p,:,:] = np.identity(mat_size, dtype=np.complex128)
                elif cascade_method == 'S':
                    matrix_group['Matrices'][p,:,:] = np.block([[np.zeros((2*n_modes, 2*n_modes),dtype=np.complex128), np.identity(2*n_modes,dtype=np.complex128)],[np.identity(2*n_modes,dtype=np.complex128), np.zeros((2*n_modes, 2*n_modes),dtype=np.complex128)]])
            
        print('Matrix arrays initialized...')
                    
        #########################
        # Begin Main Simulation #
        #########################
        
        switch = False
        switched = False

        print('Beginnig matrix cascade...')
        print('Start thickness: ' + str(dL_mfp*n_mat_array[0]))
        print('End thickness:   ' + str(dL_mfp*n_mat_array[-1]))
        
        for thickness_index, n_matrices in enumerate(n_mat_array):
            
            # Keep hdf5 file open for reading + writing data
            with h5py.File(pathname + hdf5_filename, 'r+') as f:
                matrix_group = f['Random Matrices']
                data_group = f['Data']
                
                print('{}'.format(name))
                print('Next thickness = ' + str(dL_mfp*n_matrices))            
                ############################################################
                # Go to next thickness and check for switch condition to S #
                ############################################################

                print('Performing matrix products...')
                matrices = matrix_group['Matrices']

                # First thickness: use matrices from the single pool
                if n_matrices == 1:
                    
                    if cascade_method == "M":
                        random_pool = matrix_group['Single Pool M']
                        
                        # If n_times is less than single pool size, just fill the matrix array
                        if n_times <= n_single_pool:
                            for p in range(n_times):
                                matrices[p,:,:] = random_pool[p,:,:]
                            
                        else:
                            # Fill first n_single_pool non-randomly and randomly sample the leftovers
                            for p in range(n_times):
                                if p < n_single_pool:
                                    matrices[p,:,:] = random_pool[p,:,:]
                                else:
                                    index = np.random.randint(0,n_single_pool)
                                    Mnew = random_pool[index,:,:]
                                    matrices[p,:,:] = matrices[p,:,:]@Mnew
                    
                    # Cascade method is S
                    else:

                        random_pool = matrix_group['Single Pool S']
                        
                        # If n_times is less than single pool size, just fill the matrix array
                        if n_times <= n_single_pool:
                            for p in range(n_times):
                                matrices[p,:,:] = random_pool[p,:,:]
                            
                        else:
                            # Fill first n_single_pool non-randomly and randomly sample the leftovers
                            for p in range(n_times):
                                if p < n_single_pool:
                                    matrices[p,:,:] = random_pool[p,:,:]
                                else:
                                    index = np.random.randint(0,n_single_pool)
                                    Mnew = random_pool[index,:,:]
                                    matrices[p,:,:] = Mnew

                        
                # For all thicknesses after the first one
                else:
                    if cascade_method == 'S':
                        
                        random_pool = matrix_group['Multi Pool S']
                        
                        for p in range(n_times):
                            index = np.random.randint(0,n_multi_pool)
                            S_new = random_pool[index,:,:]
                            matrices[p,:,:] = S_cascade(S_new,matrices[p,:,:],sigma_p)
                
                    elif cascade_method == 'M':
                        
                        random_pool = matrix_group['Multi Pool M']
        
                        for p in range(n_times):
                            index = np.random.randint(0,n_multi_pool)
                            M_new = random_pool[index,:,:]
                            matrices[p,:,:] = matrices[p,:,:]@M_new
                            
                            # Check if M matrices are becoming too large
                            if not switch and np.max(np.abs(matrices[p,:,:])) > 10**6:
                                print('Transfer matrix elements too large...')
                                print('Maximum value = ', np.max(np.abs(matrices[p,:,:])))
                                switch = True
                            
                # Change to S array if switch condition reached
                if switch and not switched:
                    print('Changing to S matrix cascade...')
        
                    # Convert M matrix array to S matrix array
                    for p in range(n_times):
                        matrices[p,:,:] = M2S(matrices[p,:,:])
                                            
                    switched = True
                    cascade_method = 'S'
                    with open(os.path.join(pathname,'switch.txt'), 'w') as f:
                        f.write(f'Switch occured at step {thickness_index}\n')

    
            #######################
            # Work out statistics #
            #######################
            
            # Fill in data array for matrices in each matrix array
             
            with h5py.File(pathname + hdf5_filename, 'r+') as f:
             
                print('Calculating statistics...')
                new_data = get_data(f, data_blocks, cascade_method)            

                print('Saving statistics...')
                data_group = f['Data']
                data_group['Transmission'][thickness_index] = new_data['Transmission']
                data_group['n_matrices'][thickness_index] = n_matrices
                data_group['Thicknesses'][thickness_index] = n_matrices*dL
                data_group['Tau'][thickness_index] = new_data['Tau']
               
                for mode in data_blocks:
                    mode_str = str(mode)
                    data_group[mode_str]['Diattenuation'][thickness_index,:] = new_data[mode_str]['Diattenuation']
                    data_group[mode_str]['Retardance'][thickness_index,:] = new_data[mode_str]['Retardance']
                    data_group[mode_str]['Mean'][thickness_index,:,:] = new_data[mode_str]['Mean']
                    data_group[mode_str]['H'][thickness_index,:,:] = new_data[mode_str]['H']
                    data_group[mode_str]['Matrices'][thickness_index,:,:,:] = new_data[mode_str]['Matrices']

                    
if __name__ == '__main__':
    main()