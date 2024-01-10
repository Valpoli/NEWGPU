NVCC=nvcc
NVFLAGS=-allow-unsupported-compiler

all: ex1_1 ex1_2 ex2_1 ex2_2 ex3_1 ex3_2 ex3_3


ex1_1: ex1_1.cu   ex1.cu ex1.h
	$(NVCC) $(NVFLAGS) ex1_1.cu   ex1.cu -o ex1_1
ex1_2: ex1_2.cu   ex1.cu ex1.h
	$(NVCC) $(NVFLAGS) ex1_2.cu   ex1.cu -o ex1_2

ex2_1: ex2_1.cu   ex2.cu ex2.h 
	$(NVCC) $(NVFLAGS) ex2_1.cu   ex2.cu -o ex2_1
ex2_2: ex2_2.cu   ex2.cu ex2.h 
	$(NVCC) $(NVFLAGS) ex2_2.cu   ex2.cu -o ex2_2

ex3_1: ex3_1.cu   ex3.cu ex3.h image.o
	$(NVCC) $(NVFLAGS) ex3_1.cu   ex3.cu image.o -o ex3_1
ex3_2: ex3_2.cu   ex3.cu ex3.h image.o
	$(NVCC) $(NVFLAGS) ex3_2.cu   ex3.cu image.o -o ex3_2
ex3_3: ex3_3.cu   ex3.cu ex3.h image.o
	$(NVCC) $(NVFLAGS) ex3_3.cu   ex3.cu image.o -o ex3_3


image.o: image.cpp image.h
	g++ -c image.cpp


.PHONY: clean
clean:
	rm image.o ex1_1 ex1_2 ex2_1 ex2_2 ex3_1 ex3_2 ex3_3