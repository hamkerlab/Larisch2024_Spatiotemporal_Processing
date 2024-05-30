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
// Main Structure for the population of id 3 (pop3)
///////////////////////////////////////////////////////////////
struct PopStruct3{

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

    // Global parameter v_rest
    double  v_rest ;

    // Global parameter cm
    double  cm ;

    // Global parameter tau_m
    double  tau_m ;

    // Global parameter tau_syn_E
    double  tau_syn_E ;

    // Global parameter tau_syn_I
    double  tau_syn_I ;

    // Global parameter e_rev_E
    double  e_rev_E ;

    // Global parameter e_rev_I
    double  e_rev_I ;

    // Global parameter tau_w
    double  tau_w ;

    // Global parameter a
    double  a ;

    // Global parameter b
    double  b ;

    // Global parameter gL
    double  gL ;

    // Global parameter i_offset
    double  i_offset ;

    // Global parameter delta_T
    double  delta_T ;

    // Global parameter v_thresh
    double  v_thresh ;

    // Global parameter v_reset
    double  v_reset ;

    // Global parameter v_spike
    double  v_spike ;

    // Local variable I
    std::vector< double > I;

    // Local variable v
    std::vector< double > v;

    // Local variable w
    std::vector< double > w;

    // Local variable g_exc
    std::vector< double > g_exc;

    // Local variable g_inh
    std::vector< double > g_inh;

    // Local variable r
    std::vector< double > r;

    // Random numbers


    // Delayed variables
    // Delays for spike population
    std::deque< std::vector<int> > _delayed_spike;


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

        // Local variable I
        if ( name.compare("I") == 0 ) {
            return I;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            return w;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            return g_exc;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r;
        }


        // should not happen
        std::cerr << "PopStruct3::get_local_attribute_all_double: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute_double(std::string name, int rk) {
        assert( (rk < size) );

        // Local variable I
        if ( name.compare("I") == 0 ) {
            return I[rk];
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            return v[rk];
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            return w[rk];
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            return g_exc[rk];
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            return g_inh[rk];
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            return r[rk];
        }


        // should not happen
        std::cerr << "PopStruct3::get_local_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_local_attribute_all_double(std::string name, std::vector<double> value) {
        assert( (value.size() == size) );

        // Local variable I
        if ( name.compare("I") == 0 ) {
            I = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v = value;
            return;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            w = value;
            return;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            g_exc = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct3::set_local_attribute_all_double: " << name << " not found" << std::endl;
    }

    void set_local_attribute_double(std::string name, int rk, double value) {
        assert( (rk < size) );

        // Local variable I
        if ( name.compare("I") == 0 ) {
            I[rk] = value;
            return;
        }

        // Local variable v
        if ( name.compare("v") == 0 ) {
            v[rk] = value;
            return;
        }

        // Local variable w
        if ( name.compare("w") == 0 ) {
            w[rk] = value;
            return;
        }

        // Local variable g_exc
        if ( name.compare("g_exc") == 0 ) {
            g_exc[rk] = value;
            return;
        }

        // Local variable g_inh
        if ( name.compare("g_inh") == 0 ) {
            g_inh[rk] = value;
            return;
        }

        // Local variable r
        if ( name.compare("r") == 0 ) {
            r[rk] = value;
            return;
        }


        // should not happen
        std::cerr << "PopStruct3::set_local_attribute_double: " << name << " not found" << std::endl;
    }

    double get_global_attribute_double(std::string name) {

        // Global parameter v_rest
        if ( name.compare("v_rest") == 0 ) {
            return v_rest;
        }

        // Global parameter cm
        if ( name.compare("cm") == 0 ) {
            return cm;
        }

        // Global parameter tau_m
        if ( name.compare("tau_m") == 0 ) {
            return tau_m;
        }

        // Global parameter tau_syn_E
        if ( name.compare("tau_syn_E") == 0 ) {
            return tau_syn_E;
        }

        // Global parameter tau_syn_I
        if ( name.compare("tau_syn_I") == 0 ) {
            return tau_syn_I;
        }

        // Global parameter e_rev_E
        if ( name.compare("e_rev_E") == 0 ) {
            return e_rev_E;
        }

        // Global parameter e_rev_I
        if ( name.compare("e_rev_I") == 0 ) {
            return e_rev_I;
        }

        // Global parameter tau_w
        if ( name.compare("tau_w") == 0 ) {
            return tau_w;
        }

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            return a;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            return b;
        }

        // Global parameter gL
        if ( name.compare("gL") == 0 ) {
            return gL;
        }

        // Global parameter i_offset
        if ( name.compare("i_offset") == 0 ) {
            return i_offset;
        }

        // Global parameter delta_T
        if ( name.compare("delta_T") == 0 ) {
            return delta_T;
        }

        // Global parameter v_thresh
        if ( name.compare("v_thresh") == 0 ) {
            return v_thresh;
        }

        // Global parameter v_reset
        if ( name.compare("v_reset") == 0 ) {
            return v_reset;
        }

        // Global parameter v_spike
        if ( name.compare("v_spike") == 0 ) {
            return v_spike;
        }


        // should not happen
        std::cerr << "PopStruct3::get_global_attribute_double: " << name << " not found" << std::endl;
        return static_cast<double>(0.0);
    }

    void set_global_attribute_double(std::string name, double value)  {

        // Global parameter v_rest
        if ( name.compare("v_rest") == 0 ) {
            v_rest = value;
            return;
        }

        // Global parameter cm
        if ( name.compare("cm") == 0 ) {
            cm = value;
            return;
        }

        // Global parameter tau_m
        if ( name.compare("tau_m") == 0 ) {
            tau_m = value;
            return;
        }

        // Global parameter tau_syn_E
        if ( name.compare("tau_syn_E") == 0 ) {
            tau_syn_E = value;
            return;
        }

        // Global parameter tau_syn_I
        if ( name.compare("tau_syn_I") == 0 ) {
            tau_syn_I = value;
            return;
        }

        // Global parameter e_rev_E
        if ( name.compare("e_rev_E") == 0 ) {
            e_rev_E = value;
            return;
        }

        // Global parameter e_rev_I
        if ( name.compare("e_rev_I") == 0 ) {
            e_rev_I = value;
            return;
        }

        // Global parameter tau_w
        if ( name.compare("tau_w") == 0 ) {
            tau_w = value;
            return;
        }

        // Global parameter a
        if ( name.compare("a") == 0 ) {
            a = value;
            return;
        }

        // Global parameter b
        if ( name.compare("b") == 0 ) {
            b = value;
            return;
        }

        // Global parameter gL
        if ( name.compare("gL") == 0 ) {
            gL = value;
            return;
        }

        // Global parameter i_offset
        if ( name.compare("i_offset") == 0 ) {
            i_offset = value;
            return;
        }

        // Global parameter delta_T
        if ( name.compare("delta_T") == 0 ) {
            delta_T = value;
            return;
        }

        // Global parameter v_thresh
        if ( name.compare("v_thresh") == 0 ) {
            v_thresh = value;
            return;
        }

        // Global parameter v_reset
        if ( name.compare("v_reset") == 0 ) {
            v_reset = value;
            return;
        }

        // Global parameter v_spike
        if ( name.compare("v_spike") == 0 ) {
            v_spike = value;
            return;
        }


        std::cerr << "PopStruct3::set_global_attribute_double: " << name << " not found" << std::endl;
    }



    // Method called to initialize the data structures
    void init_population() {
    #ifdef _DEBUG
        std::cout << "PopStruct3::init_population(size="<<this->size<<") - this = " << this << std::endl;
    #endif
        _active = true;

        // Global parameter v_rest
        v_rest = 0.0;

        // Global parameter cm
        cm = 0.0;

        // Global parameter tau_m
        tau_m = 0.0;

        // Global parameter tau_syn_E
        tau_syn_E = 0.0;

        // Global parameter tau_syn_I
        tau_syn_I = 0.0;

        // Global parameter e_rev_E
        e_rev_E = 0.0;

        // Global parameter e_rev_I
        e_rev_I = 0.0;

        // Global parameter tau_w
        tau_w = 0.0;

        // Global parameter a
        a = 0.0;

        // Global parameter b
        b = 0.0;

        // Global parameter gL
        gL = 0.0;

        // Global parameter i_offset
        i_offset = 0.0;

        // Global parameter delta_T
        delta_T = 0.0;

        // Global parameter v_thresh
        v_thresh = 0.0;

        // Global parameter v_reset
        v_reset = 0.0;

        // Global parameter v_spike
        v_spike = 0.0;

        // Local variable I
        I = std::vector<double>(size, 0.0);

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable w
        w = std::vector<double>(size, 0.0);

        // Local variable g_exc
        g_exc = std::vector<double>(size, 0.0);

        // Local variable g_inh
        g_inh = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);


        // Spiking variables
        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);


        // Delayed variables
        _delayed_spike = std::deque< std::vector<int> >(max_delay, std::vector<int>());

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


        _delayed_spike.clear();
        _delayed_spike = std::deque< std::vector<int> >(max_delay, std::vector<int>());

    }

    // Method to draw new random numbers
    void update_rng() {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct3::update_rng()" << std::endl;
#endif

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

        if ( _active ) {

            _delayed_spike.push_front(spiked);
            _delayed_spike.pop_back();

        }
    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

        if(value <= max_delay){ // nothing to do
            return;
        }
        max_delay = value;

        _delayed_spike.resize(max_delay, std::vector<int>());

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {



            // Updating the step sizes
            double __stepsize_g_exc = 1.0 - exp( -dt/tau_syn_E);
            double __stepsize_g_inh = 1.0 - exp( -dt/tau_syn_I);

            // Updating local variables
            #pragma omp simd
            for(int i = 0; i < size; i++){

                // I = g_exc - g_inh + i_offset
                I[i] = g_exc[i] - g_inh[i] + i_offset;


                // dv/dt = ( -gL*(v-v_rest) + gL*delta_T*exp((v-v_thresh)/delta_T) + I - w)/cm
                double _v = (I[i] + delta_T*gL*exp((v[i] - v_thresh)/delta_T) - gL*v[i] + gL*v_rest - w[i])/cm;

                // tau_w * dw/dt = a * (v - v_rest) - w
                double _w = (a*v[i] - a*v_rest - w[i])/tau_w;

                // tau_syn_E * dg_exc/dt = - g_exc
                double _g_exc =  __stepsize_g_exc*(0 - g_exc[i]);

                // tau_syn_I * dg_inh/dt = - g_inh
                double _g_inh =  __stepsize_g_inh*(0 - g_inh[i]);

                // dv/dt = ( -gL*(v-v_rest) + gL*delta_T*exp((v-v_thresh)/delta_T) + I - w)/cm
                v[i] += dt*_v ;


                // tau_w * dw/dt = a * (v - v_rest) - w
                w[i] += dt*_w ;


                // tau_syn_E * dg_exc/dt = - g_exc
                g_exc[i] += _g_exc ;


                // tau_syn_I * dg_inh/dt = - g_inh
                g_inh[i] += _g_inh ;


            }
        } // active

    }

    void spike_gather() {

        if( _active ) {
            spiked.clear();

            for (int i = 0; i < size; i++) {


                // Spike emission
                if(v[i] >= v_spike){ // Condition is met
                    // Reset variables

                    v[i] = v_reset;

                    w[i] += b;

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
        size_in_bytes += sizeof(double);	// v_rest
        size_in_bytes += sizeof(double);	// cm
        size_in_bytes += sizeof(double);	// tau_m
        size_in_bytes += sizeof(double);	// tau_syn_E
        size_in_bytes += sizeof(double);	// tau_syn_I
        size_in_bytes += sizeof(double);	// e_rev_E
        size_in_bytes += sizeof(double);	// e_rev_I
        size_in_bytes += sizeof(double);	// tau_w
        size_in_bytes += sizeof(double);	// a
        size_in_bytes += sizeof(double);	// b
        size_in_bytes += sizeof(double);	// gL
        size_in_bytes += sizeof(double);	// i_offset
        size_in_bytes += sizeof(double);	// delta_T
        size_in_bytes += sizeof(double);	// v_thresh
        size_in_bytes += sizeof(double);	// v_reset
        size_in_bytes += sizeof(double);	// v_spike
        // Variables
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * I.capacity();	// I
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * w.capacity();	// w
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_exc.capacity();	// g_exc
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * g_inh.capacity();	// g_inh
        size_in_bytes += sizeof(std::vector<double>) + sizeof(double) * r.capacity();	// r
        // RNGs

        return size_in_bytes;
    }

    // Memory management: destroy all the C++ data
    void clear() {
#ifdef _DEBUG
    std::cout << "PopStruct3::clear() - this = " << this << std::endl;
#endif

            #ifdef _DEBUG
                std::cout << "PopStruct3::clear()" << std::endl;
            #endif
        // Parameters

        // Variables
        I.clear();
        I.shrink_to_fit();
        v.clear();
        v.shrink_to_fit();
        w.clear();
        w.shrink_to_fit();
        g_exc.clear();
        g_exc.shrink_to_fit();
        g_inh.clear();
        g_inh.shrink_to_fit();
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

    }
};

