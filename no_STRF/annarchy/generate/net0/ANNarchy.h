#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <random>
#include <cassert>
// only included if compiled with -fopenmp
#ifdef _OPENMP
    #include <omp.h>
#endif

// Intrinsic operations (Intel/AMD)
#ifdef __x86_64__
    #include <immintrin.h>
#endif

/*
 * Built-in functions
 *
 */

#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) % long(b)
#define ite(a, b, c) (a?b:c)

// power function for integer exponent
inline double power(double x, unsigned int a){
    double res=x;
    for (unsigned int i=0; i< a-1; i++){
        res *= x;
    }
    return res;
};


/*
 * Custom constants
 *
 */


/*
 * Custom functions
 *
 */


/*
 * Structures for the populations
 *
 */
#include "pop0.hpp"
#include "pop1.hpp"
#include "pop2.hpp"
#include "pop3.hpp"
#include "pop4.hpp"
#include "pop5.hpp"
#include "pop6.hpp"

/*
 * Structures for the projections
 *
 */
#include "proj0.hpp"
#include "proj1.hpp"
#include "proj2.hpp"
#include "proj3.hpp"
#include "proj4.hpp"
#include "proj5.hpp"
#include "proj6.hpp"
#include "proj7.hpp"
#include "proj8.hpp"
#include "proj9.hpp"
#include "proj10.hpp"


/*
 * Declaration of the populations
 *
 */
extern PopStruct0 pop0;
extern PopStruct1 pop1;
extern PopStruct2 pop2;
extern PopStruct3 pop3;
extern PopStruct4 pop4;
extern PopStruct5 pop5;
extern PopStruct6 pop6;


/*
 * Declaration of the projections
 *
 */
extern ProjStruct0 proj0;
extern ProjStruct1 proj1;
extern ProjStruct2 proj2;
extern ProjStruct3 proj3;
extern ProjStruct4 proj4;
extern ProjStruct5 proj5;
extern ProjStruct6 proj6;
extern ProjStruct7 proj7;
extern ProjStruct8 proj8;
extern ProjStruct9 proj9;
extern ProjStruct10 proj10;


/*
 * Recorders
 *
 */
#include "Recorder.h"

extern std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(Monitor* recorder);

/*
 * Simulation methods
 *
 */
void run(const int nbSteps);
int run_until(const int steps, std::vector<int> populations, bool or_and);
void step();

/*
 *  Initialization
 */
void initialize(const double dt_) ;

/*
 *  Life-time management
 */
void create_cpp_instances();
void destroy_cpp_instances();

/*
 * Time export
 *
 */
long int getTime();
void setTime(const long int t_);
double getDt();
void setDt(const double dt_);

/*
 * Number of threads
 *
 */
void setNumberThreads(int threads, std::vector<int> core_list);

/*
 * Seed for the RNG
 *
*/
void setSeed(long int seed, int num_sources, bool use_seed_seq);

