# FFTW
CPP = g++
FFTW3_DIR = /sw/keeneland/fftw/3.2.1/centos5.4_gnu4.1.2_fPIC
CFLAGS += -I$(FFTW3_DIR)/include
LDFLAGS += -L$(FFTW3_DIR)/lib -lfftw3_threads -lfftw3 -lpthread -lgomp

#MKL New Style
#CPP=icpc
#MKLROOT=/opt/intel/composer_xe_2011_sp1.8.273/mkl
#CFLAGS += -I$(MKLROOT)/include/fftw -O2
#LDFLAGS = -mkl=parallel

# CUFFT (CUDA FFT)
#NVCC = nvcc
#NVCCFLAGS = -O2 -arch sm_20
#LDFLAGS += -lcufft

fft: fft.cpp
	$(CPP) $(CFLAGS) fft.cpp -o fft $(LDFLAGS)

cufft: cufft.cu
	$(NVCC) $(NVCCFLAGS) cufft.cu -o cufft $(LDFLAGS)

clean:
	rm fft
	rm cufft

new:
	make clean
	make
