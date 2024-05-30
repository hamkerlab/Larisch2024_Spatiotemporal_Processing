/*
 *  ANNarchy-version: 4.7.3
 */
#pragma once

#include "ANNarchy.h"
#include <random>



extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 6 (I1)
///////////////////////////////////////////////////////////////
struct PopStruct6{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }



    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Neuron specific parameters and variables

    // Local parameter gL
    std::vector< double > gL;

    // Local parameter DeltaT
    std::vector< double > DeltaT;

    // Local parameter tauw
    std::vector< double > tauw;

    // Local parameter a
    std::vector< double > a;

    // Local parameter b
    std::vector< double > b;

    // Local parameter EL
    std::vector< double > EL;

    // Local parameter C
    std::vector< double > C;

    // Local parameter tauz
    std::vector< double > tauz;

    // Local parameter tauVT
    std::vector< double > tauVT;

    // Local parameter Isp
    std::vector< double > Isp;

    // Local parameter VTMax
    std::vector< double > VTMax;

    // Local parameter VTrest
    std::vector< double > VTrest;

    // Local parameter taux
    std::vector< double > taux;

    // Local parameter tauLTD
    std::vector< double > tauLTD;

    // Local parameter tauLTP
    std::vector< double > tauLTP;

    // Local parameter taumean
    std::vector< double > taumean;

    // Local parameter tau_gExc
    std::vector< double > tau_gExc;

    // Local parameter tau_gInh
    std::vector< double > tau_gInh;

    // Local variable noise
    std::vector< double > noise;

    // Local variable vm
    std::vector< double > vm;

    // Local variable vmean
    std::vector< double > vmean;

    // Local variable umeanLTD
    std::vector< double > umeanLTD;

    // Local variable umeanLTP
    std::vector< double > umeanLTP;

    // Local variable xtrace
    std::vector< double > xtrace;

    // Local variable wad
    std::vector< double > wad;

    // Local variable z
    std::vector< double > z;

    // Local variable VT
    std::vector< double > VT;

    // Local variable g_Exc
    std::vector< double > g_Exc;

    // Local variable g_Inh
    std::vector< double > g_Inh;

    // Local variable state
    std::vector< double > state;

    // Local variable Spike
    std::vector< double > Spike;

    // Local variable resetvar
    std::vector< double > resetvar;

    // Local variable vmTemp
    std::vector< double > vmTemp;

    // Local variable r
    std::vector< double > r;

    // Random numbers
std::vector<double> rand_0 ;


    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate(double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
            if (_spike_history.empty())
                _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        }
    };


    // Access methods to the parameters and variables

    std::vector<double> get_local_attribute_all_double(std::string name) {

        // Local parameter gL
        if ( name.compare("gL") == 0 ) {
            return gL;
        }

        // Local parameter DeltaT
        if ( name.compare("DeltaT") == 0 ) {
            return DeltaT;
        }

        // Local parameter tauw
        if ( name.compare("tauw") == 0 ) {
            return tauw;
        }

        // Local parameter a
        if ( name.compare("a") == 0 ) {
            return a;
        }

        // Local parameter b
        if ( name.compare("b") == 0 ) {
            return b;
        }

        // Local parameter EL
        if ( name.compare("EL") == 0 ) {
            return EL;
        }

        // Local parameter C
        if ( name.compare("C") == 0 ) {
            return C;
        }

        // Local parameter tauz
        if ( name.compare("tauz") == 0 ) {
            return tauz;
        }

        // Local parameter tauVT
        if ( name.compare("tauVT") == 0 ) {
            return tauVT;
        }

        // Local parameter Isp
        if ( name.compare("Isp") == 0 ) {
            return Isp;
        }

        // Local parameter VTMax
        if ( name.compare("VTMax") == 0 ) {
            return VTMax;
        }

        // Local parameter VTrest
        if ( name.compare("VTrest") == 0 ) {
            return VTrest;
        }

        // Local parameter taux
        if ( name.compare("taux") == 0 ) {
            return taux;
        }

        // Local parameter tauLTD
        if ( name.compare("tauLTD") == 0 ) {
            return tauLTD;
        }

        // Local parameter tauLTP
        if ( name.compare("tauLTP") == 0 ) {
            return tauLTP;
        }

        // Local parameter taumean
        if ( name.compare("taumean") == 0 ) {
            return taumean;
        }

        // Local parameter tau_gExc
        if ( name.compare("tau_gExc") == 0 ) {
            return tau_gExc;
        }

        // Local parameter tau_gInh
        if ( name.compare("tau_gInh") == 0 ) {
            return tau_gInh;
        }

        // Local variable noise
        if ( name.compare("noise") == 0 ) {
            return noise;
        }

        // Local variable vm
        if ( name.compare("vm") == 0 ) {
            return vm;
        }

        // Local variable vmean
        if ( name.compare("vmean") == 0 ) {
            return vmean;
        }

        // Local variable umeanLTD
        if ( name.compare("umeanLTD") == 0 ) {
            return umeanLTD;
        }

        // Local variable umeanLTP
        if ( name.compare("umeanLTP") == 0 ) {
            return umeanLTP;
        }

        // Local variable xtrace
        if ( name.compare("xtrace") == 0 ) {
            return xtrace;
        }

        // Local variable wad
        if ( name.compare("wad") == 0 ) {
            return wad;
        }

        // Local variable z
        if ( name.compare("z") == 0 ) {
            return z;
        }

        // Local variable VT
        if ( name.compare("VT") == 0 ) {
            return VT;
        }

        // Local variable g_Exc
        if ( name.compare("g_Exc") == 0 ) {
            return g_Exc;
        }

        // Local variable g_Inh
        if ( name.compare("g_Inh") == 0 ) {
            return g_Inh;
        }

        // Local variable state
        if ( name.compare("state") == 0 ) {
            return state;
        }

        // Local variable Spike
        if ( name.compare("Spike") == 0 ) {
            return Spike;
        }

        // Local variable resetvar
        if ( name.compare("resetvar") == 0 ) {
            return resetvar;
        }

        // Local variable vmTemp
        if ( name.compare("vmTemp") == 0 ) {
            return vmTemp;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct6::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local parameter gL
        if ( name.compare("gL") == 0 ) {
            return gL[rk];
        }

        // Local parameter DeltaT
        if ( name.compare("DeltaT") == 0 ) {
            return DeltaT[rk];
        }

        // Local parameter tauw
        if ( name.compare("tauw") == 0 ) {
            return tauw[rk];
        }

        // Local parameter a
        if ( name.compare("a") == 0 ) {
            return a[rk];
        }

        // Local parameter b
        if ( name.compare("b") == 0 ) {
            return b[rk];
        }

        // Local parameter EL
        if ( name.compare("EL") == 0 ) {
            return EL[rk];
        }

        // Local parameter C
        if ( name.compare("C") == 0 ) {
            return C[rk];
        }

        // Local parameter tauz
        if ( name.compare("tauz") == 0 ) {
            return tauz[rk];
        }

        // Local parameter tauVT
        if ( name.compare("tauVT") == 0 ) {
            return tauVT[rk];
        }

        // Local parameter Isp
        if ( name.compare("Isp") == 0 ) {
            return Isp[rk];
        }

        // Local parameter VTMax
        if ( name.compare("VTMax") == 0 ) {
            return VTMax[rk];
        }

        // Local parameter VTrest
        if ( name.compare("VTrest") == 0 ) {
            return VTrest[rk];
        }

        // Local parameter taux
        if ( name.compare("taux") == 0 ) {
            return taux[rk];
        }

        // Local parameter tauLTD
        if ( name.compare("tauLTD") == 0 ) {
            return tauLTD[rk];
        }

        // Local parameter tauLTP
        if ( name.compare("tauLTP") == 0 ) {
            return tauLTP[rk];
        }

        // Local parameter taumean
        if ( name.compare("taumean") == 0 ) {
            return taumean[rk];
        }

        // Local parameter tau_gExc
        if ( name.compare("tau_gExc") == 0 ) {
            return tau_gExc[rk];
        }

        // Local parameter tau_gInh
        if ( name.compare("tau_gInh") == 0 ) {
            return tau_gInh[rk];
        }

        // Local variable noise
        if ( name.compare("noise") == 0 ) {
            return noise[rk];
        }

        // Local variable vm
        if ( name.compare("vm") == 0 ) {
            return vm[rk];
        }

        // Local variable vmean
        if ( name.compare("vmean") == 0 ) {
            return vmean[rk];
        }

        // Local variable umeanLTD
        if ( name.compare("umeanLTD") == 0 ) {
            return umeanLTD[rk];
        }

        // Local variable umeanLTP
        if ( name.compare("umeanLTP") == 0 ) {
            return umeanLTP[rk];
        }

        // Local variable xtrace
        if ( name.compare("xtrace") == 0 ) {
            return xtrace[rk];
        }

        // Local variable wad
        if ( name.compare("wad") == 0 ) {
            return wad[rk];
        }

        // Local variable z
        if ( name.compare("z") == 0 ) {
            return z[rk];
        }

        // Local variable VT
        if ( name.compare("VT") == 0 ) {
            return VT[rk];
        }

        // Local variable g_Exc
        if ( name.compare("g_Exc") == 0 ) {
            return g_Exc[rk];
        }

        // Local variable g_Inh
        if ( name.compare("g_Inh") == 0 ) {
            return g_Inh[rk];
        }

        // Local variable state
        if ( name.compare("state") == 0 ) {
            return state[rk];
        }

        // Local variable Spike
        if ( name.compare("Spike") == 0 ) {
            return Spike[rk];
        }

        // Local variable resetvar
        if ( name.compare("resetvar") == 0 ) {
            return resetvar[rk];
        }

        // Local variable vmTemp
        if ( name.compare("vmTemp") == 0 ) {
            return vmTemp[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct6::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local parameter gL
        if ( name.compare("gL") == 0 ) {
            gL = value;
            return;
        }

        // Local parameter DeltaT
        if ( name.compare("DeltaT") == 0 ) {
            DeltaT = value;
            return;
        }

        // Local parameter tauw
        if ( name.compare("tauw") == 0 ) {
            tauw = value;
            return;
        }

        // Local parameter a
        if ( name.compare("a") == 0 ) {
            a = value;
            return;
        }

        // Local parameter b
        if ( name.compare("b") == 0 ) {
            b = value;
            return;
        }

        // Local parameter EL
        if ( name.compare("EL") == 0 ) {
            EL = value;
            return;
        }

        // Local parameter C
        if ( name.compare("C") == 0 ) {
            C = value;
            return;
        }

        // Local parameter tauz
        if ( name.compare("tauz") == 0 ) {
            tauz = value;
            return;
        }

        // Local parameter tauVT
        if ( name.compare("tauVT") == 0 ) {
            tauVT = value;
            return;
        }

        // Local parameter Isp
        if ( name.compare("Isp") == 0 ) {
            Isp = value;
            return;
        }

        // Local parameter VTMax
        if ( name.compare("VTMax") == 0 ) {
            VTMax = value;
            return;
        }

        // Local parameter VTrest
        if ( name.compare("VTrest") == 0 ) {
            VTrest = value;
            return;
        }

        // Local parameter taux
        if ( name.compare("taux") == 0 ) {
            taux = value;
            return;
        }

        // Local parameter tauLTD
        if ( name.compare("tauLTD") == 0 ) {
            tauLTD = value;
            return;
        }

        // Local parameter tauLTP
        if ( name.compare("tauLTP") == 0 ) {
            tauLTP = value;
            return;
        }

        // Local parameter taumean
        if ( name.compare("taumean") == 0 ) {
            taumean = value;
            return;
        }

        // Local parameter tau_gExc
        if ( name.compare("tau_gExc") == 0 ) {
            tau_gExc = value;
            return;
        }

        // Local parameter tau_gInh
        if ( name.compare("tau_gInh") == 0 ) {
            tau_gInh = value;
            return;
        }

        // Local variable noise
        if ( name.compare("noise") == 0 ) {
            noise = value;
            return;
        }

        // Local variable vm
        if ( name.compare("vm") == 0 ) {
            vm = value;
            return;
        }

        // Local variable vmean
        if ( name.compare("vmean") == 0 ) {
            vmean = value;
            return;
        }

        // Local variable umeanLTD
        if ( name.compare("umeanLTD") == 0 ) {
            umeanLTD = value;
            return;
        }

        // Local variable umeanLTP
        if ( name.compare("umeanLTP") == 0 ) {
            umeanLTP = value;
            return;
        }

        // Local variable xtrace
        if ( name.compare("xtrace") == 0 ) {
            xtrace = value;
            return;
        }

        // Local variable wad
        if ( name.compare("wad") == 0 ) {
            wad = value;
            return;
        }

        // Local variable z
        if ( name.compare("z") == 0 ) {
            z = value;
            return;
        }

        // Local variable VT
        if ( name.compare("VT") == 0 ) {
            VT = value;
            return;
        }

        // Local variable g_Exc
        if ( name.compare("g_Exc") == 0 ) {
            g_Exc = value;
            return;
        }

        // Local variable g_Inh
        if ( name.compare("g_Inh") == 0 ) {
            g_Inh = value;
            return;
        }

        // Local variable state
        if ( name.compare("state") == 0 ) {
            state = value;
            return;
        }

        // Local variable Spike
        if ( name.compare("Spike") == 0 ) {
            Spike = value;
            return;
        }

        // Local variable resetvar
        if ( name.compare("resetvar") == 0 ) {
            resetvar = value;
            return;
        }

        // Local variable vmTemp
        if ( name.compare("vmTemp") == 0 ) {
            vmTemp = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct6::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local parameter gL
        if ( name.compare("gL") == 0 ) {
            gL[rk] = value;
            return;
        }

        // Local parameter DeltaT
        if ( name.compare("DeltaT") == 0 ) {
            DeltaT[rk] = value;
            return;
        }

        // Local parameter tauw
        if ( name.compare("tauw") == 0 ) {
            tauw[rk] = value;
            return;
        }

        // Local parameter a
        if ( name.compare("a") == 0 ) {
            a[rk] = value;
            return;
        }

        // Local parameter b
        if ( name.compare("b") == 0 ) {
            b[rk] = value;
            return;
        }

        // Local parameter EL
        if ( name.compare("EL") == 0 ) {
            EL[rk] = value;
            return;
        }

        // Local parameter C
        if ( name.compare("C") == 0 ) {
            C[rk] = value;
            return;
        }

        // Local parameter tauz
        if ( name.compare("tauz") == 0 ) {
            tauz[rk] = value;
            return;
        }

        // Local parameter tauVT
        if ( name.compare("tauVT") == 0 ) {
            tauVT[rk] = value;
            return;
        }

        // Local parameter Isp
        if ( name.compare("Isp") == 0 ) {
            Isp[rk] = value;
            return;
        }

        // Local parameter VTMax
        if ( name.compare("VTMax") == 0 ) {
            VTMax[rk] = value;
            return;
        }

        // Local parameter VTrest
        if ( name.compare("VTrest") == 0 ) {
            VTrest[rk] = value;
            return;
        }

        // Local parameter taux
        if ( name.compare("taux") == 0 ) {
            taux[rk] = value;
            return;
        }

        // Local parameter tauLTD
        if ( name.compare("tauLTD") == 0 ) {
            tauLTD[rk] = value;
            return;
        }

        // Local parameter tauLTP
        if ( name.compare("tauLTP") == 0 ) {
            tauLTP[rk] = value;
            return;
        }

        // Local parameter taumean
        if ( name.compare("taumean") == 0 ) {
            taumean[rk] = value;
            return;
        }

        // Local parameter tau_gExc
        if ( name.compare("tau_gExc") == 0 ) {
            tau_gExc[rk] = value;
            return;
        }

        // Local parameter tau_gInh
        if ( name.compare("tau_gInh") == 0 ) {
            tau_gInh[rk] = value;
            return;
        }

        // Local variable noise
        if ( name.compare("noise") == 0 ) {
            noise[rk] = value;
            return;
        }

        // Local variable vm
        if ( name.compare("vm") == 0 ) {
            vm[rk] = value;
            return;
        }

        // Local variable vmean
        if ( name.compare("vmean") == 0 ) {
            vmean[rk] = value;
            return;
        }

        // Local variable umeanLTD
        if ( name.compare("umeanLTD") == 0 ) {
            umeanLTD[rk] = value;
            return;
        }

        // Local variable umeanLTP
        if ( name.compare("umeanLTP") == 0 ) {
            umeanLTP[rk] = value;
            return;
        }

        // Local variable xtrace
        if ( name.compare("xtrace") == 0 ) {
            xtrace[rk] = value;
            return;
        }

        // Local variable wad
        if ( name.compare("wad") == 0 ) {
            wad[rk] = value;
            return;
        }

        // Local variable z
        if ( name.compare("z") == 0 ) {
            z[rk] = value;
            return;
        }

        // Local variable VT
        if ( name.compare("VT") == 0 ) {
            VT[rk] = value;
            return;
        }

        // Local variable g_Exc
        if ( name.compare("g_Exc") == 0 ) {
            g_Exc[rk] = value;
            return;
        }

        // Local variable g_Inh
        if ( name.compare("g_Inh") == 0 ) {
            g_Inh[rk] = value;
            return;
        }

        // Local variable state
        if ( name.compare("state") == 0 ) {
            state[rk] = value;
            return;
        }

        // Local variable Spike
        if ( name.compare("Spike") == 0 ) {
            Spike[rk] = value;
            return;
        }

        // Local variable resetvar
        if ( name.compare("resetvar") == 0 ) {
            resetvar[rk] = value;
            return;
        }

        // Local variable vmTemp
        if ( name.compare("vmTemp") == 0 ) {
            vmTemp[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct6::set_local_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct6::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Local parameter gL
        gL = std::vector<double>(size, 0.0);

        // Local parameter DeltaT
        DeltaT = std::vector<double>(size, 0.0);

        // Local parameter tauw
        tauw = std::vector<double>(size, 0.0);

        // Local parameter a
        a = std::vector<double>(size, 0.0);

        // Local parameter b
        b = std::vector<double>(size, 0.0);

        // Local parameter EL
        EL = std::vector<double>(size, 0.0);

        // Local parameter C
        C = std::vector<double>(size, 0.0);

        // Local parameter tauz
        tauz = std::vector<double>(size, 0.0);

        // Local parameter tauVT
        tauVT = std::vector<double>(size, 0.0);

        // Local parameter Isp
        Isp = std::vector<double>(size, 0.0);

        // Local parameter VTMax
        VTMax = std::vector<double>(size, 0.0);

        // Local parameter VTrest
        VTrest = std::vector<double>(size, 0.0);

        // Local parameter taux
        taux = std::vector<double>(size, 0.0);

        // Local parameter tauLTD
        tauLTD = std::vector<double>(size, 0.0);

        // Local parameter tauLTP
        tauLTP = std::vector<double>(size, 0.0);

        // Local parameter taumean
        taumean = std::vector<double>(size, 0.0);

        // Local parameter tau_gExc
        tau_gExc = std::vector<double>(size, 0.0);

        // Local parameter tau_gInh
        tau_gInh = std::vector<double>(size, 0.0);

        // Local variable noise
        noise = std::vector<double>(size, 0.0);

        // Local variable vm
        vm = std::vector<double>(size, 0.0);

        // Local variable vmean
        vmean = std::vector<double>(size, 0.0);

        // Local variable umeanLTD
        umeanLTD = std::vector<double>(size, 0.0);

        // Local variable umeanLTP
        umeanLTP = std::vector<double>(size, 0.0);

        // Local variable xtrace
        xtrace = std::vector<double>(size, 0.0);

        // Local variable wad
        wad = std::vector<double>(size, 0.0);

        // Local variable z
        z = std::vector<double>(size, 0.0);

        // Local variable VT
        VT = std::vector<double>(size, 0.0);

        // Local variable g_Exc
        g_Exc = std::vector<double>(size, 0.0);

        // Local variable g_Inh
        g_Inh = std::vector<double>(size, 0.0);

        // Local variable state
        state = std::vector<double>(size, 0.0);

        // Local variable Spike
        Spike = std::vector<double>(size, 0.0);

        // Local variable resetvar
        resetvar = std::vector<double>(size, 0.0);

        // Local variable vmTemp
        vmTemp = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);

        // Random numbers
        rand_0 = std::vector<double>(size, 0.0);

        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >();
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        // Spiking variables
        spiked.clear();
        spiked.shrink_to_fit();
        std::fill(last_spike.begin(), last_spike.end(), -10000L);

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            if (!it->empty()) {
                auto empty_queue = std::queue<long int>();
                it->swap(empty_queue);
            }
        }



    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct6::update_rng()" << std::endl;
#endif

        if (_active) {
auto dist_rand_0 = std::normal_distribution< double >(0.0, 1.0);

            for(int i = 0; i < size; i++) {
rand_0[i] = dist_rand_0(rng[0]);
            }
        }

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {



            // Updating local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // noise = Normal(0.0,1.0)
                noise[i] = rand_0[i];


                // dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc-g_Inh
                double _vm = (state[i] >= 2 ? 3.4620000000000002 : (state[i] == 1 ? g_Exc[i] - g_Inh[i] - vm[i] - 51.75 + 1*(Isp[i] - (b[i] + wad[i]))/C[i] : g_Exc[i] - g_Inh[i] + 1*(DeltaT[i]*gL[i]*exp((-VT[i] + vm[i])/DeltaT[i]) + (-gL[i])*(-EL[i] + vm[i]) - wad[i] + z[i])/C[i]));

                // dvmean/dt = ((vm - EL)**2 - vmean)/taumean
                double _vmean = (-vmean[i] + pow(EL[i] - vm[i], 2))/taumean[i];

                // dumeanLTD/dt = (vm - umeanLTD)/tauLTD
                double _umeanLTD = (-umeanLTD[i] + vm[i])/tauLTD[i];

                // dumeanLTP/dt = (vm - umeanLTP)/tauLTP
                double _umeanLTP = (-umeanLTP[i] + vm[i])/tauLTP[i];

                // dxtrace /dt = (- xtrace )/taux
                double _xtrace = -xtrace[i]/taux[i];

                // dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw
                double _wad = (state[i] == 2 ? 0 : (state[i] == 1 ? b[i]/tauw[i] : (a[i]*(-EL[i] + vm[i]) - wad[i])/tauw[i]));

                // dz/dt = if state==1:-z+Isp-10 else:-z/tauz
                double _z = (state[i] == 1 ? Isp[i] - z[i] - 10 : (-z[i])/tauz[i]);

                // dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT
                double _VT = (state[i] == 1 ? -VT[i] + VTMax[i] - 0.40000000000000002 : (-VT[i] + VTrest[i])/tauVT[i]);

                // dg_Exc/dt = 1/tau_gExc * (-g_Exc)
                double _g_Exc = -g_Exc[i]/tau_gExc[i];

                // dg_Inh/dt = 1/tau_gInh*(-g_Inh)
                double _g_Inh = -g_Inh[i]/tau_gInh[i];

                // dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc-g_Inh
                vm[i] += dt*_vm ;


                // dvmean/dt = ((vm - EL)**2 - vmean)/taumean
                vmean[i] += dt*_vmean ;


                // dumeanLTD/dt = (vm - umeanLTD)/tauLTD
                umeanLTD[i] += dt*_umeanLTD ;


                // dumeanLTP/dt = (vm - umeanLTP)/tauLTP
                umeanLTP[i] += dt*_umeanLTP ;


                // dxtrace /dt = (- xtrace )/taux
                xtrace[i] += dt*_xtrace ;


                // dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw
                wad[i] += dt*_wad ;


                // dz/dt = if state==1:-z+Isp-10 else:-z/tauz
                z[i] += dt*_z ;


                // dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT
                VT[i] += dt*_VT ;


                // dg_Exc/dt = 1/tau_gExc * (-g_Exc)
                g_Exc[i] += dt*_g_Exc ;


                // dg_Inh/dt = 1/tau_gInh*(-g_Inh)
                g_Inh[i] += dt*_g_Inh ;


                // state = if state > 0: state-1 else:0
                state[i] = (state[i] > 0 ? state[i] - 1 : 0);


                // Spike = 0.0
                Spike[i] = 0.0;


                // dresetvar / dt = 1/(1.0) * (-resetvar)
                double _resetvar = -1.0*resetvar[i];

                // dresetvar / dt = 1/(1.0) * (-resetvar)
                resetvar[i] += dt*_resetvar ;


                // vmTemp = vm
                vmTemp[i] = vm[i];


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            spiked.clear();

            for (int i = 0; i < size; i++) {


                // Spike emission
                if(state[i] == 0.0 && vm[i] > VT[i]){ // Condition is met
                    // Reset variables

                    vm[i] = 29.399999999999999;

                    state[i] = 2.0;

                    Spike[i] = 1.0;

                    resetvar[i] = 1.0;

                    xtrace[i] += 1.0/taux[i];

                    // Store the spike
                    spiked.push_back(i);
                    last_spike[i] = t;

                    // Refractory period


                    // Store the event for the mean firing rate
                    if (_mean_fr_window > 0)
                        _spike_history[i].push(t);

                }

            }

            // Update mean firing rate
            if (_mean_fr_window > 0) {
                for (int i = 0; i < size; i++) {
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * double(_spike_history[i].size());
                }
            }
        } // active

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * gL.capacity();	// gL
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * DeltaT.capacity();	// DeltaT
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tauw.capacity();	// tauw
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * a.capacity();	// a
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * b.capacity();	// b
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * EL.capacity();	// EL
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * C.capacity();	// C
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tauz.capacity();	// tauz
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tauVT.capacity();	// tauVT
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Isp.capacity();	// Isp
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * VTMax.capacity();	// VTMax
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * VTrest.capacity();	// VTrest
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * taux.capacity();	// taux
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tauLTD.capacity();	// tauLTD
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tauLTP.capacity();	// tauLTP
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * taumean.capacity();	// taumean
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tau_gExc.capacity();	// tau_gExc
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * tau_gInh.capacity();	// tau_gInh
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * noise.capacity();	// noise
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * vm.capacity();	// vm
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * vmean.capacity();	// vmean
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * umeanLTD.capacity();	// umeanLTD
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * umeanLTP.capacity();	// umeanLTP
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * xtrace.capacity();	// xtrace
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * wad.capacity();	// wad
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * z.capacity();	// z
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * VT.capacity();	// VT
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_Exc.capacity();	// g_Exc
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_Inh.capacity();	// g_Inh
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * state.capacity();	// state
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * Spike.capacity();	// Spike
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * resetvar.capacity();	// resetvar
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * vmTemp.capacity();	// vmTemp
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * rand_0.capacity();	// rand_0

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct6::clear() - this = " << this << std::endl;
#endif

            #ifdef _DEBUG
                std::cout << "PopStruct6::clear()" << std::endl;
            #endif
        // Parameters
        gL.clear();
        gL.shrink_to_fit();
        DeltaT.clear();
        DeltaT.shrink_to_fit();
        tauw.clear();
        tauw.shrink_to_fit();
        a.clear();
        a.shrink_to_fit();
        b.clear();
        b.shrink_to_fit();
        EL.clear();
        EL.shrink_to_fit();
        C.clear();
        C.shrink_to_fit();
        tauz.clear();
        tauz.shrink_to_fit();
        tauVT.clear();
        tauVT.shrink_to_fit();
        Isp.clear();
        Isp.shrink_to_fit();
        VTMax.clear();
        VTMax.shrink_to_fit();
        VTrest.clear();
        VTrest.shrink_to_fit();
        taux.clear();
        taux.shrink_to_fit();
        tauLTD.clear();
        tauLTD.shrink_to_fit();
        tauLTP.clear();
        tauLTP.shrink_to_fit();
        taumean.clear();
        taumean.shrink_to_fit();
        tau_gExc.clear();
        tau_gExc.shrink_to_fit();
        tau_gInh.clear();
        tau_gInh.shrink_to_fit();

        // Variables
        noise.clear();
        noise.shrink_to_fit();
        vm.clear();
        vm.shrink_to_fit();
        vmean.clear();
        vmean.shrink_to_fit();
        umeanLTD.clear();
        umeanLTD.shrink_to_fit();
        umeanLTP.clear();
        umeanLTP.shrink_to_fit();
        xtrace.clear();
        xtrace.shrink_to_fit();
        wad.clear();
        wad.shrink_to_fit();
        z.clear();
        z.shrink_to_fit();
        VT.clear();
        VT.shrink_to_fit();
        g_Exc.clear();
        g_Exc.shrink_to_fit();
        g_Inh.clear();
        g_Inh.shrink_to_fit();
        state.clear();
        state.shrink_to_fit();
        Spike.clear();
        Spike.shrink_to_fit();
        resetvar.clear();
        resetvar.shrink_to_fit();
        vmTemp.clear();
        vmTemp.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();

        // Spike events
        spiked.clear();
        spiked.shrink_to_fit();

        last_spike.clear();
        last_spike.shrink_to_fit();

        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            while(!it->empty())
                it->pop();
        }
        _spike_history.clear();
        _spike_history.shrink_to_fit();

        // RNGs
        rand_0.clear();
        rand_0.shrink_to_fit();
    }
};

