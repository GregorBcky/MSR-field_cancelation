# MSR-field_cancelation

This repository shall be the backbone of the bachelor thesis regarding the field cancelation coils of the magnetic shield room (_MSR_).

## Overview of files and their purpose

- ```MSR_map_coil_config.ipynb```: Created by Chiara Weckmann. Only comments and markdowns were added. This is the originally program, everything is based on.
- ```Versuch_1_Auswertung.ipynb```: Created simulate effect of basic (circular) coil setups. General workflow: Known coil-layup -> resulting magnetic field.
This programm makes heavy use of ```Externe_Classes.py``` and ```Ausgelagerte_Funktionen_Versuchsauswertung.py```. Results are stored in ```B_tot_results```. The directory ```B_tot_results_square_MSR``` holds old results for different geometry of the MSR. Inputs are stored in the directory ```Example_map``` and ```Experiments```. Where both background fields and fields with current are stored.
- ```Auswertung_npz_Dateien.ipynb``` makes multiple plots to the result of ```Versuch_1_Auswertung.ipynb```. The data is not changed here.
- ```Custom_B_field.ipynb``` creates a directory ```Desired_map``` containing the B-field which was inputted either by inputting the $B$-field directly or its vector potential $A$. Within the directory, the defined input is stored, such that the program can extract the data analogously to the MSR-data.
- ```Field_cancelation.ipynb``` takes the background field as input and creates a coil-layup to cancel this field. It creates the ```Layup.pdf``` file, showing the layup of coils of all faces. It also creates the directory ```Contours``` which holds .txt files of the contours, which than can be loaded into Autodesk-Fusion using the ```ImportContoursFromTxt.py``` programm which is stored in the ```ImportContoursFromTxt``` directory.
- ```Ausgelagerte_Funktionen_Versuchsauswertung.py``` holds all the functions, which are just in this project. It is mainly used in ```Field_cancelation.ipynb``` and ```Versuch_1_Auswertung.ipynb```.
- ```Externe_Classes.py``` holds two classes for the coil-plane-layup and meshing as well as the $\mu$-metal meshing. It is mainly used by ```Versuch_1_Auswertung.ipynb``` and in ```Field_cancelation.ipynb```.
- ```ImportContoursFromTxt.py``` is located in the ```ImportContoursFromTxt``` directory and loads the contour lines which are stored in the directory ```Contours``` into Autodesk-Fusion. This program needs to be started from Autodesk-Fusion!
