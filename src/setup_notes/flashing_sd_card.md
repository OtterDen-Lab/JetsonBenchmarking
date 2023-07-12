Installation instructions for the Jetson Orin Nano

- In order to have all packages that might be needed, install JetPack. I will be doing the SD Card image method that contains the OS to run the device and JetPack.

- Go to 'https://developer.nvidia.com/embedded/jetpack' and scroll down to *SD Card Image Method*. Select JETSON Orin Nano DEVELOPER KIT. It will download a .img file that will be used to flash the sd card.

- Go for a walk. The download might take a while, it is 22.1gb file as of writing this.

- Once download is complete, download a image writer. I will be using Balena Etcher. It can be found at this webpage 'https://etcher.balena.io/#download-etcher'. Scroll down until you see downloads. For me, I will be downloading the Linux x64 appimage, but install what you need for your system.

- Install or open Balena Etcher. For step one, select the .img file that we downloaded earlier. After that, select the sd card we are installing to. Finally, confirm the select is correct and flash. 

- Congragulations, you have flashed an OS to the sd card for your Jetson Orin Nano. Once the flash is complete, remove the sd card and insert it into your Jetson. The card slot in on the bottom of the gpu model. The sd card must face up when inserting. Now boot up your Jetson and enjoy.
