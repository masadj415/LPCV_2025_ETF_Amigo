# kako prebaciti dataset na lokalnoj mrezi

prebacivanje dataseta na lokalnoj mrezi moguce je sa scp (seccure copy protocol) komandom

https://www.geeksforgeeks.org/scp-command-in-linux-with-examples/

scp [options] [[user@]host1:]source_file_or_directory ... [[user@]host2:]destination

npr za kopiranje datasets foldera treba napisati

opcija -r rekurzivno kopira sve fajlove koji su u folderu na zadatom source_directory

scp -r centar15-node1@10.118.5.150:/home/centar15-node1/LPCV_2025_T1/datasets /home/centar15-desktop1/LPCV_2025_T1/datasets/

