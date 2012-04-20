#include <iostream>
#include <fftw3.h>
#include <stddef.h>
#include <sys/time.h>

using namespace std;

double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main(int argc, char* argv[]) {
    
    // Double Complex
    fftw_complex *in, *out;
    fftw_plan fwdPlan, invPlan;
    
    if (argc != 2) { 
        cerr << "usage: ./fft N, where N*1024 is number of elems" << endl;
        exit(-1);
    }
    int count = atoi(argv[1]) * 1024;

    in = new fftw_complex[count];
    out = new fftw_complex[count];

    //Copies original real data, set to zero imaginary part
    for(int i = 0; i < count; ++i) {
        //real number
        in[i][0] = 12;
        //i - number
        in[i][1] = 0;

    }
    fftw_plan_with_nthreads(12);
    fwdPlan = fftw_plan_dft_1d(count, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwdPlan);
    
    double start = second();
    for (int i = 0; i < 100; i ++) {
        fftw_execute(fwdPlan);
    }
    double stop = second();
    cout << count << ", " << (stop - start) / 100.0 << endl;

    delete [] in; delete [] out;
    fftw_destroy_plan(fwdPlan);
    
}
