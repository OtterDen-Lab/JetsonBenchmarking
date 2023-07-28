Instructions for installing Tensorflow

Step 1.) Open a terminal on your Jetson Orin Nano. This can be through ssh.

Step 2.) Type into that terminal 
		'''sudo apt-get update
		sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran'''
		
Step 3.) Install and upgrade 'pip3'
		'''sudo apt-get install python3-pip
		sudo python3 -m pip install --upgrade pip
		sudo pip3 install -U testresources setuptools==65.5.0'''
		
Step 4.) Install the Python package dependencies

	'sudo pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.6.0'
	
Step 5.) Install Tensorflow

	'sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v511 tensorflow==2.12.0+nv23.05'
