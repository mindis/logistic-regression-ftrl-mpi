#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I/opt/OpenBLAS/include
LIBRARYPATH = -L/usr/local/lib -L/opt/OpenBLAS/lib 
#LIBRARY = -lboost_thread -lboost_system -lpthread -lglog -lm
LIBRARY = -lpthread -lm
CPP_tag = -std=gnu++11

lr_ftrl_mpi: main.o
	mpicxx $(CPP_tag) -o lr_ftrl_mpi main.o $(LIBRARYPATH) $(LIBRARY) -lopenblas 

main.o: src/main.cpp
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

clean:
	rm -f *~ lr_ftrl_mpi predict train_ut *.o
