This is the setup file for setting up ssh on Jetson Orin Nano and your host machine. This file currently assumes you are running linux on both devices. This file does not contain windows, andriod, or apple instructions.

Step 1.) Verify that ssh is installed on both devices.

	- Open a terminal
	
	- Type 'ssh' into the terminal and press enter.
	
		-- You should recieve
		
		'''usage: ssh [-1246AaCfGgKkMNnqsTtVvXxYy] [-b bind_address] [-c cipher_spec] [-D [bind_address:]port] [-E log_file] [-e escape_char] [-F configfile] [-I pkcs11] [-i identity_file] [-J [user@]host[:port]] [-L address] [-l login_name] [-m mac_spec] [-O ctl_cmd] [-o option] [-p port] [-Q query_option] [-R address] [-S ctl_path] [-W host:port] [-w local_tun[:remote_tun]] [user@]hostname [command]'''
		
		-- If you don't have ssh installed, you will need to install it. Type 'sudo apt-get install openssh-client' into the terminal and follow the onscreen prompts to install the package.
		
		--Type 'ssh' into the terminal to verify the package was installed correctly.
		
	- (For the Orin Nano) Type 'ssh localhost' into the terminal and press enter.
	You should recieve '''The authenticitu of host 'localhost (xxx.x.x.x)' can't be established.
	ECDSA key fingerprint is ...
	Are you sure you want to continue connecting (yes,no,[fingerprint])?'''.
	
	-If you get that message, then you have the SSH server package installed and don't need to follow the next sub-steps.
	
		-- Type 'sudo apt-get install openssh-server' into the terminal to install the SSH server package and then follow the on-screen prompts to install the package.
		
		-- Type 'ssh localhost' to verify that the package installed correctly.
		
Step 2.) Get the ip address of your Jetson Orin Nano.

	- Type 'hostname -I' into your Jetson's terminal. The first string of numbers will be the IP address we need.
	
Step 3.) Access the Jetson through ssh

	- Type 'ssh -X [hostname]@[IP address]' into a terminal on your laptop. From there, you will have to accept the SSH fingerprint and then enter the password to your Jetson. Once completed, you will be able to access your Jetson through SSH and open some GUI programs on your laptop through the command line.


	
	
