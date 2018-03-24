NVCC = /usr/local/cuda-9.1/bin/nvcc
all: main run clean git
main:	main.o
	$(NVCC) main.o -o main
main.o:	main.cu
	$(NVCC) main.cu -c  main.o -arch=sm_30 --compiler-options -fPIC -shared
.PHONY:	clean run
clean:
	-rm main main.o
run:
	cuda-memcheck ./main
git:
	git add .; git commit -m "implemented"; git push;
