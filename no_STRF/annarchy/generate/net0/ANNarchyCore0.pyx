# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from libcpp.string cimport string
from math import ceil
import numpy as np
import sys
cimport numpy as np
cimport cython

# Short names for unsigned integer types
ctypedef unsigned char _ann_uint8
ctypedef unsigned short _ann_uint16
ctypedef unsigned int _ann_uint32
ctypedef unsigned long _ann_uint64

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL

cdef extern from "ANNarchy.h":

    # User-defined functions


    # User-defined constants


    # Data structures

    # Export Population 0 (pop0)
    cdef struct PopStruct0 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)





        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 1 (pop1)
    cdef struct PopStruct1 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)





        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 2 (pop2)
    cdef struct PopStruct2 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)





        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 3 (pop3)
    cdef struct PopStruct3 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 4 (pop4)
    cdef struct PopStruct4 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)

        # Global attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 5 (E1)
    cdef struct PopStruct5 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 6 (I1)
    cdef struct PopStruct6 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local attributes
        vector[double] get_local_attribute_all_double(string)
        double get_local_attribute_double(string, int)
        void set_local_attribute_all_double(string, vector[double])
        void set_local_attribute_double(string, int, double)



        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()


    # Export Projection 0
    cdef struct ProjStruct0 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 1
    cdef struct ProjStruct1 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 2
    cdef struct ProjStruct2 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)





        # Global Attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 3
    cdef struct ProjStruct3 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)





        # Global Attributes
        double get_global_attribute_double(string)
        void set_global_attribute_double(string, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 4
    cdef struct ProjStruct4 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 5
    cdef struct ProjStruct5 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 6
    cdef struct ProjStruct6 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 7
    cdef struct ProjStruct7 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()



        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()



        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 8
    cdef struct ProjStruct8 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()





        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 9
    cdef struct ProjStruct9 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()





        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 10
    cdef struct ProjStruct10 :
        # Flags
        bool _transmission
        bool _axon_transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset

        # Connectivity
        bool init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[ vector[int] ] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)
        int nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)

        map[int, int] nb_efferent_synapses()





        # Local Attributes
        vector[vector[double]] get_local_attribute_all_double(string)
        vector[double] get_local_attribute_row_double(string, int)
        double get_local_attribute_double(string, int, int)
        void set_local_attribute_all_double(string, vector[vector[double]])
        void set_local_attribute_row_double(string, int, vector[double])
        void set_local_attribute_double(string, int, int, double)





        # cuda configuration


        # memory management
        long int size_in_bytes()
        void clear()



    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_


    # Population 0 (pop0) : Monitor
    cdef cppclass PopRecorder0 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder0* get_instance(int)
        long int size_in_bytes()
        void clear()

        # Targets
    # Population 1 (pop1) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder1* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] r
        bool record_r

        # Targets
        vector[vector[double]] _sum_exc
        bool record__sum_exc

    # Population 2 (pop2) : Monitor
    cdef cppclass PopRecorder2 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder2* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] r
        bool record_r

        # Targets
        vector[vector[double]] _sum_exc
        bool record__sum_exc

    # Population 3 (pop3) : Monitor
    cdef cppclass PopRecorder3 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder3* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] w
        bool record_w

        vector[vector[double]] g_exc
        bool record_g_exc

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 4 (pop4) : Monitor
    cdef cppclass PopRecorder4 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder4* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] I
        bool record_I

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] w
        bool record_w

        vector[vector[double]] g_exc
        bool record_g_exc

        vector[vector[double]] g_inh
        bool record_g_inh

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 5 (E1) : Monitor
    cdef cppclass PopRecorder5 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder5* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] noise
        bool record_noise

        vector[vector[double]] vm
        bool record_vm

        vector[vector[double]] vmean
        bool record_vmean

        vector[vector[double]] umeanLTD
        bool record_umeanLTD

        vector[vector[double]] umeanLTP
        bool record_umeanLTP

        vector[vector[double]] xtrace
        bool record_xtrace

        vector[vector[double]] wad
        bool record_wad

        vector[vector[double]] z
        bool record_z

        vector[vector[double]] VT
        bool record_VT

        vector[vector[double]] g_Exc
        bool record_g_Exc

        vector[vector[double]] g_Inh
        bool record_g_Inh

        vector[vector[double]] state
        bool record_state

        vector[vector[double]] Spike
        bool record_Spike

        vector[vector[double]] resetvar
        bool record_resetvar

        vector[vector[double]] vmTemp
        bool record_vmTemp

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 6 (I1) : Monitor
    cdef cppclass PopRecorder6 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder6* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] noise
        bool record_noise

        vector[vector[double]] vm
        bool record_vm

        vector[vector[double]] vmean
        bool record_vmean

        vector[vector[double]] umeanLTD
        bool record_umeanLTD

        vector[vector[double]] umeanLTP
        bool record_umeanLTP

        vector[vector[double]] xtrace
        bool record_xtrace

        vector[vector[double]] wad
        bool record_wad

        vector[vector[double]] z
        bool record_z

        vector[vector[double]] VT
        bool record_VT

        vector[vector[double]] g_Exc
        bool record_g_Exc

        vector[vector[double]] g_Inh
        bool record_g_Inh

        vector[vector[double]] state
        bool record_state

        vector[vector[double]] Spike
        bool record_Spike

        vector[vector[double]] resetvar
        bool record_resetvar

        vector[vector[double]] vmTemp
        bool record_vmTemp

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Projection 0 : Monitor
    cdef cppclass ProjRecorder0 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder0* get_instance(int)

    # Projection 1 : Monitor
    cdef cppclass ProjRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder1* get_instance(int)

    # Projection 2 : Monitor
    cdef cppclass ProjRecorder2 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder2* get_instance(int)

    # Projection 3 : Monitor
    cdef cppclass ProjRecorder3 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder3* get_instance(int)

    # Projection 4 : Monitor
    cdef cppclass ProjRecorder4 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder4* get_instance(int)

    # Projection 5 : Monitor
    cdef cppclass ProjRecorder5 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder5* get_instance(int)

    # Projection 6 : Monitor
    cdef cppclass ProjRecorder6 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder6* get_instance(int)

    # Projection 7 : Monitor
    cdef cppclass ProjRecorder7 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder7* get_instance(int)

    # Projection 8 : Monitor
    cdef cppclass ProjRecorder8 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder8* get_instance(int)

    # Projection 9 : Monitor
    cdef cppclass ProjRecorder9 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder9* get_instance(int)

    # Projection 10 : Monitor
    cdef cppclass ProjRecorder10 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder10* get_instance(int)


    # Instances

    PopStruct0 pop0
    PopStruct1 pop1
    PopStruct2 pop2
    PopStruct3 pop3
    PopStruct4 pop4
    PopStruct5 pop5
    PopStruct6 pop6

    ProjStruct0 proj0
    ProjStruct1 proj1
    ProjStruct2 proj2
    ProjStruct3 proj3
    ProjStruct4 proj4
    ProjStruct5 proj5
    ProjStruct6 proj6
    ProjStruct7 proj7
    ProjStruct8 proj8
    ProjStruct9 proj9
    ProjStruct10 proj10

    # Methods
    void create_cpp_instances()
    void initialize(double)
    void destroy_cpp_instances()
    void setSeed(long, int, bool)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)


    # Number of threads
    void setNumberThreads(int, vector[int])


# Profiling (if needed)


# Population wrappers

# Wrapper for population 0 (pop0)
@cython.auto_pickle(True)
cdef class pop0_wrapper :

    def __init__(self, size, max_delay):

        pop0.set_size(size)
        pop0.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop0.get_size()
    # Reset the population
    def reset(self):
        pop0.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop0.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop0.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop0.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop0.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop0.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop0.set_local_attribute_double(cpp_string, rk, value)







    # memory management
    def size_in_bytes(self):
        return pop0.size_in_bytes()

    def clear(self):
        return pop0.clear()

# Wrapper for population 1 (pop1)
@cython.auto_pickle(True)
cdef class pop1_wrapper :

    def __init__(self, size, max_delay):

        pop1.set_size(size)
        pop1.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop1.get_size()
    # Reset the population
    def reset(self):
        pop1.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop1.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop1.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop1.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop1.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop1.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop1.set_local_attribute_double(cpp_string, rk, value)







    # memory management
    def size_in_bytes(self):
        return pop1.size_in_bytes()

    def clear(self):
        return pop1.clear()

# Wrapper for population 2 (pop2)
@cython.auto_pickle(True)
cdef class pop2_wrapper :

    def __init__(self, size, max_delay):

        pop2.set_size(size)
        pop2.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop2.get_size()
    # Reset the population
    def reset(self):
        pop2.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop2.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop2.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop2.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop2.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop2.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop2.set_local_attribute_double(cpp_string, rk, value)







    # memory management
    def size_in_bytes(self):
        return pop2.size_in_bytes()

    def clear(self):
        return pop2.clear()

# Wrapper for population 3 (pop3)
@cython.auto_pickle(True)
cdef class pop3_wrapper :

    def __init__(self, size, max_delay):

        pop3.set_size(size)
        pop3.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop3.get_size()
    # Reset the population
    def reset(self):
        pop3.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop3.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop3.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop3.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop3.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop3.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop3.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop3.set_global_attribute_double(cpp_string, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop3.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop3.size_in_bytes()

    def clear(self):
        return pop3.clear()

# Wrapper for population 4 (pop4)
@cython.auto_pickle(True)
cdef class pop4_wrapper :

    def __init__(self, size, max_delay):

        pop4.set_size(size)
        pop4.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop4.get_size()
    # Reset the population
    def reset(self):
        pop4.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop4.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop4.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop4.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop4.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop4.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_local_attribute_double(cpp_string, rk, value)


    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop4.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop4.set_global_attribute_double(cpp_string, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop4.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop4.size_in_bytes()

    def clear(self):
        return pop4.clear()

# Wrapper for population 5 (E1)
@cython.auto_pickle(True)
cdef class pop5_wrapper :

    def __init__(self, size, max_delay):

        pop5.set_size(size)
        pop5.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop5.get_size()
    # Reset the population
    def reset(self):
        pop5.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop5.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop5.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop5.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop5.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop5.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop5.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop5.set_local_attribute_double(cpp_string, rk, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop5.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop5.size_in_bytes()

    def clear(self):
        return pop5.clear()

# Wrapper for population 6 (I1)
@cython.auto_pickle(True)
cdef class pop6_wrapper :

    def __init__(self, size, max_delay):

        pop6.set_size(size)
        pop6.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop6.get_size()
    # Reset the population
    def reset(self):
        pop6.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop6.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop6.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop6.set_active(val)


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return np.array(pop6.get_local_attribute_all_double(cpp_string))


    def get_local_attribute(self, name, rk, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return pop6.get_local_attribute_double(cpp_string, rk)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop6.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute(self, name, rk, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            pop6.set_local_attribute_double(cpp_string, rk, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop6.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop6.size_in_bytes()

    def clear(self):
        return pop6.clear()


# Projection wrappers

# Wrapper for projection 0
@cython.auto_pickle(True)
cdef class proj0_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj0.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj0.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj0._transmission
    def _set_transmission(self, bool l):
        proj0._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj0._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj0._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj0._update
    def _set_update(self, bool l):
        proj0._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj0._plasticity
    def _set_plasticity(self, bool l):
        proj0._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj0._update_period
    def _set_update_period(self, int l):
        proj0._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj0._update_offset
    def _set_update_offset(self, long l):
        proj0._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj0.nb_dendrites()

    def post_rank(self):
        return proj0.get_post_rank()
    def pre_rank_all(self):
        return proj0.get_pre_ranks()
    def pre_rank(self, int n):
        return proj0.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj0.nb_dendrites()
    def nb_synapses(self):
        return proj0.nb_synapses()
    def dendrite_size(self, int n):
        return proj0.dendrite_size(n)



    # Access to non-uniform delay
    def get_delay(self):
        return proj0.delay
    def get_dendrite_delay(self, idx):
        return proj0.delay[idx]
    def set_delay(self, value):
        proj0.delay = value
    def get_max_delay(self):
        return proj0.max_delay
    def set_max_delay(self, value):
        proj0.max_delay = value
    def update_max_delay(self, value):
        proj0.update_max_delay(value)
    def reset_ring_buffer(self):
        proj0.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj0.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj0.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj0.size_in_bytes()

    def clear(self):
        return proj0.clear()

# Wrapper for projection 1
@cython.auto_pickle(True)
cdef class proj1_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj1.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj1.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj1._transmission
    def _set_transmission(self, bool l):
        proj1._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj1._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj1._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj1._update
    def _set_update(self, bool l):
        proj1._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj1._plasticity
    def _set_plasticity(self, bool l):
        proj1._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj1._update_period
    def _set_update_period(self, int l):
        proj1._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj1._update_offset
    def _set_update_offset(self, long l):
        proj1._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj1.nb_dendrites()

    def post_rank(self):
        return proj1.get_post_rank()
    def pre_rank_all(self):
        return proj1.get_pre_ranks()
    def pre_rank(self, int n):
        return proj1.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj1.nb_dendrites()
    def nb_synapses(self):
        return proj1.nb_synapses()
    def dendrite_size(self, int n):
        return proj1.dendrite_size(n)



    # Access to non-uniform delay
    def get_delay(self):
        return proj1.delay
    def get_dendrite_delay(self, idx):
        return proj1.delay[idx]
    def set_delay(self, value):
        proj1.delay = value
    def get_max_delay(self):
        return proj1.max_delay
    def set_max_delay(self, value):
        proj1.max_delay = value
    def update_max_delay(self, value):
        proj1.update_max_delay(value)
    def reset_ring_buffer(self):
        proj1.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj1.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj1.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj1.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj1.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj1.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj1.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj1.size_in_bytes()

    def clear(self):
        return proj1.clear()

# Wrapper for projection 2
@cython.auto_pickle(True)
cdef class proj2_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj2.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj2.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj2._transmission
    def _set_transmission(self, bool l):
        proj2._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj2._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj2._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj2._update
    def _set_update(self, bool l):
        proj2._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj2._plasticity
    def _set_plasticity(self, bool l):
        proj2._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj2._update_period
    def _set_update_period(self, int l):
        proj2._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj2._update_offset
    def _set_update_offset(self, long l):
        proj2._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj2.nb_dendrites()

    def post_rank(self):
        return proj2.get_post_rank()
    def pre_rank_all(self):
        return proj2.get_pre_ranks()
    def pre_rank(self, int n):
        return proj2.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj2.nb_dendrites()
    def nb_synapses(self):
        return proj2.nb_synapses()
    def dendrite_size(self, int n):
        return proj2.dendrite_size(n)




    # Global Attributes
    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":            
            return proj2.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":            
            proj2.set_global_attribute_double(cpp_string, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj2.size_in_bytes()

    def clear(self):
        return proj2.clear()

# Wrapper for projection 3
@cython.auto_pickle(True)
cdef class proj3_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj3.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj3.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj3._transmission
    def _set_transmission(self, bool l):
        proj3._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj3._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj3._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj3._update
    def _set_update(self, bool l):
        proj3._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj3._plasticity
    def _set_plasticity(self, bool l):
        proj3._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj3._update_period
    def _set_update_period(self, int l):
        proj3._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj3._update_offset
    def _set_update_offset(self, long l):
        proj3._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj3.nb_dendrites()

    def post_rank(self):
        return proj3.get_post_rank()
    def pre_rank_all(self):
        return proj3.get_pre_ranks()
    def pre_rank(self, int n):
        return proj3.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj3.nb_dendrites()
    def nb_synapses(self):
        return proj3.nb_synapses()
    def dendrite_size(self, int n):
        return proj3.dendrite_size(n)




    # Global Attributes
    def get_global_attribute(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":            
            return proj3.get_global_attribute_double(cpp_string)


    def set_global_attribute(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":            
            proj3.set_global_attribute_double(cpp_string, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj3.size_in_bytes()

    def clear(self):
        return proj3.clear()

# Wrapper for projection 4
@cython.auto_pickle(True)
cdef class proj4_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj4.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj4.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj4._transmission
    def _set_transmission(self, bool l):
        proj4._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj4._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj4._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj4._update
    def _set_update(self, bool l):
        proj4._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj4._plasticity
    def _set_plasticity(self, bool l):
        proj4._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj4._update_period
    def _set_update_period(self, int l):
        proj4._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj4._update_offset
    def _set_update_offset(self, long l):
        proj4._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj4.nb_dendrites()

    def post_rank(self):
        return proj4.get_post_rank()
    def pre_rank_all(self):
        return proj4.get_pre_ranks()
    def pre_rank(self, int n):
        return proj4.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj4.nb_dendrites()
    def nb_synapses(self):
        return proj4.nb_synapses()
    def dendrite_size(self, int n):
        return proj4.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj4.nb_efferent_synapses()



    # Access to non-uniform delay
    def get_delay(self):
        return proj4.delay
    def get_dendrite_delay(self, idx):
        return proj4.delay[idx]
    def set_delay(self, value):
        proj4.delay = value
    def get_max_delay(self):
        return proj4.max_delay
    def set_max_delay(self, value):
        proj4.max_delay = value
    def update_max_delay(self, value):
        proj4.update_max_delay(value)
    def reset_ring_buffer(self):
        proj4.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj4.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj4.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj4.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj4.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj4.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj4.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj4.size_in_bytes()

    def clear(self):
        return proj4.clear()

# Wrapper for projection 5
@cython.auto_pickle(True)
cdef class proj5_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj5.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj5.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj5._transmission
    def _set_transmission(self, bool l):
        proj5._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj5._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj5._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj5._update
    def _set_update(self, bool l):
        proj5._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj5._plasticity
    def _set_plasticity(self, bool l):
        proj5._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj5._update_period
    def _set_update_period(self, int l):
        proj5._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj5._update_offset
    def _set_update_offset(self, long l):
        proj5._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj5.nb_dendrites()

    def post_rank(self):
        return proj5.get_post_rank()
    def pre_rank_all(self):
        return proj5.get_pre_ranks()
    def pre_rank(self, int n):
        return proj5.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj5.nb_dendrites()
    def nb_synapses(self):
        return proj5.nb_synapses()
    def dendrite_size(self, int n):
        return proj5.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj5.nb_efferent_synapses()



    # Access to non-uniform delay
    def get_delay(self):
        return proj5.delay
    def get_dendrite_delay(self, idx):
        return proj5.delay[idx]
    def set_delay(self, value):
        proj5.delay = value
    def get_max_delay(self):
        return proj5.max_delay
    def set_max_delay(self, value):
        proj5.max_delay = value
    def update_max_delay(self, value):
        proj5.update_max_delay(value)
    def reset_ring_buffer(self):
        proj5.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj5.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj5.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj5.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj5.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj5.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj5.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj5.size_in_bytes()

    def clear(self):
        return proj5.clear()

# Wrapper for projection 6
@cython.auto_pickle(True)
cdef class proj6_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj6.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj6.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj6._transmission
    def _set_transmission(self, bool l):
        proj6._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj6._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj6._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj6._update
    def _set_update(self, bool l):
        proj6._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj6._plasticity
    def _set_plasticity(self, bool l):
        proj6._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj6._update_period
    def _set_update_period(self, int l):
        proj6._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj6._update_offset
    def _set_update_offset(self, long l):
        proj6._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj6.nb_dendrites()

    def post_rank(self):
        return proj6.get_post_rank()
    def pre_rank_all(self):
        return proj6.get_pre_ranks()
    def pre_rank(self, int n):
        return proj6.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj6.nb_dendrites()
    def nb_synapses(self):
        return proj6.nb_synapses()
    def dendrite_size(self, int n):
        return proj6.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj6.nb_efferent_synapses()



    # Access to non-uniform delay
    def get_delay(self):
        return proj6.delay
    def get_dendrite_delay(self, idx):
        return proj6.delay[idx]
    def set_delay(self, value):
        proj6.delay = value
    def get_max_delay(self):
        return proj6.max_delay
    def set_max_delay(self, value):
        proj6.max_delay = value
    def update_max_delay(self, value):
        proj6.update_max_delay(value)
    def reset_ring_buffer(self):
        proj6.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj6.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj6.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj6.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj6.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj6.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj6.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj6.size_in_bytes()

    def clear(self):
        return proj6.clear()

# Wrapper for projection 7
@cython.auto_pickle(True)
cdef class proj7_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj7.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj7.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj7._transmission
    def _set_transmission(self, bool l):
        proj7._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj7._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj7._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj7._update
    def _set_update(self, bool l):
        proj7._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj7._plasticity
    def _set_plasticity(self, bool l):
        proj7._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj7._update_period
    def _set_update_period(self, int l):
        proj7._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj7._update_offset
    def _set_update_offset(self, long l):
        proj7._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj7.nb_dendrites()

    def post_rank(self):
        return proj7.get_post_rank()
    def pre_rank_all(self):
        return proj7.get_pre_ranks()
    def pre_rank(self, int n):
        return proj7.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj7.nb_dendrites()
    def nb_synapses(self):
        return proj7.nb_synapses()
    def dendrite_size(self, int n):
        return proj7.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj7.nb_efferent_synapses()



    # Access to non-uniform delay
    def get_delay(self):
        return proj7.delay
    def get_dendrite_delay(self, idx):
        return proj7.delay[idx]
    def set_delay(self, value):
        proj7.delay = value
    def get_max_delay(self):
        return proj7.max_delay
    def set_max_delay(self, value):
        proj7.max_delay = value
    def update_max_delay(self, value):
        proj7.update_max_delay(value)
    def reset_ring_buffer(self):
        proj7.reset_ring_buffer()


    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj7.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj7.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj7.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj7.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj7.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj7.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj7.size_in_bytes()

    def clear(self):
        return proj7.clear()

# Wrapper for projection 8
@cython.auto_pickle(True)
cdef class proj8_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj8.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj8.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj8._transmission
    def _set_transmission(self, bool l):
        proj8._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj8._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj8._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj8._update
    def _set_update(self, bool l):
        proj8._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj8._plasticity
    def _set_plasticity(self, bool l):
        proj8._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj8._update_period
    def _set_update_period(self, int l):
        proj8._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj8._update_offset
    def _set_update_offset(self, long l):
        proj8._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj8.nb_dendrites()

    def post_rank(self):
        return proj8.get_post_rank()
    def pre_rank_all(self):
        return proj8.get_pre_ranks()
    def pre_rank(self, int n):
        return proj8.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj8.nb_dendrites()
    def nb_synapses(self):
        return proj8.nb_synapses()
    def dendrite_size(self, int n):
        return proj8.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj8.nb_efferent_synapses()




    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj8.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj8.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj8.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj8.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj8.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj8.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj8.size_in_bytes()

    def clear(self):
        return proj8.clear()

# Wrapper for projection 9
@cython.auto_pickle(True)
cdef class proj9_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj9.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj9.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj9._transmission
    def _set_transmission(self, bool l):
        proj9._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj9._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj9._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj9._update
    def _set_update(self, bool l):
        proj9._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj9._plasticity
    def _set_plasticity(self, bool l):
        proj9._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj9._update_period
    def _set_update_period(self, int l):
        proj9._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj9._update_offset
    def _set_update_offset(self, long l):
        proj9._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj9.nb_dendrites()

    def post_rank(self):
        return proj9.get_post_rank()
    def pre_rank_all(self):
        return proj9.get_pre_ranks()
    def pre_rank(self, int n):
        return proj9.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj9.nb_dendrites()
    def nb_synapses(self):
        return proj9.nb_synapses()
    def dendrite_size(self, int n):
        return proj9.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj9.nb_efferent_synapses()




    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj9.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj9.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj9.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj9.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj9.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj9.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj9.size_in_bytes()

    def clear(self):
        return proj9.clear()

# Wrapper for projection 10
@cython.auto_pickle(True)
cdef class proj10_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj10.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)

    def init_from_lil(self, post_rank, pre_rank, w, delay):
        return proj10.init_from_lil(post_rank, pre_rank, w, delay)


    # Transmission flag
    def _get_transmission(self):
        return proj10._transmission
    def _set_transmission(self, bool l):
        proj10._transmission = l

    # Transmission flag (axon spikes)
    def _get_axon_transmission(self):
        return proj10._axon_transmission
    def _set_axon_transmission(self, bool l):
        proj10._axon_transmission = l

    # Update flag
    def _get_update(self):
        return proj10._update
    def _set_update(self, bool l):
        proj10._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj10._plasticity
    def _set_plasticity(self, bool l):
        proj10._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj10._update_period
    def _set_update_period(self, int l):
        proj10._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj10._update_offset
    def _set_update_offset(self, long l):
        proj10._update_offset = l

    # Access connectivity

    property size:
        def __get__(self):
            return proj10.nb_dendrites()

    def post_rank(self):
        return proj10.get_post_rank()
    def pre_rank_all(self):
        return proj10.get_pre_ranks()
    def pre_rank(self, int n):
        return proj10.get_dendrite_pre_rank(n)
    def nb_dendrites(self):
        return proj10.nb_dendrites()
    def nb_synapses(self):
        return proj10.nb_synapses()
    def dendrite_size(self, int n):
        return proj10.dendrite_size(n)

    def nb_efferent_synapses(self):
        return proj10.nb_efferent_synapses()




    # Local Attribute
    def get_local_attribute_all(self, name, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj10.get_local_attribute_all_double(cpp_string)


    def get_local_attribute_row(self, name, rk_post, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj10.get_local_attribute_row_double(cpp_string, rk_post)


    def get_local_attribute(self, name, rk_post, rk_pre, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            return proj10.get_local_attribute_double(cpp_string, rk_post, rk_pre)


    def set_local_attribute_all(self, name, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj10.set_local_attribute_all_double(cpp_string, value)


    def set_local_attribute_row(self, name, rk_post, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj10.set_local_attribute_row_double(cpp_string, rk_post, value)


    def set_local_attribute(self, name, rk_post, rk_pre, value, ctype):
        cpp_string = name.encode('utf-8')

        if ctype == "double":
            proj10.set_local_attribute_double(cpp_string, rk_post, rk_pre, value)






        # cuda configuration


    # memory management
    def size_in_bytes(self):
        return proj10.size_in_bytes()

    def clear(self):
        return proj10.clear()


# Monitor wrappers

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder0.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder0.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder0.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).period_offset_ = val

    # Targets
# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder1.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder1.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder1.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).period_offset_ = val

    property r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder1.get_instance(self.id)).r.clear()

    # Targets
    property _sum_exc:
        def __get__(self): return (PopRecorder1.get_instance(self.id))._sum_exc
        def __set__(self, val): (PopRecorder1.get_instance(self.id))._sum_exc = val
    property record__sum_exc:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record__sum_exc
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record__sum_exc = val
    def clear__sum_exc(self):
        (PopRecorder1.get_instance(self.id))._sum_exc.clear()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder2_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder2.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder2.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder2.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).period_offset_ = val

    property r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder2.get_instance(self.id)).r.clear()

    # Targets
    property _sum_exc:
        def __get__(self): return (PopRecorder2.get_instance(self.id))._sum_exc
        def __set__(self, val): (PopRecorder2.get_instance(self.id))._sum_exc = val
    property record__sum_exc:
        def __get__(self): return (PopRecorder2.get_instance(self.id)).record__sum_exc
        def __set__(self, val): (PopRecorder2.get_instance(self.id)).record__sum_exc = val
    def clear__sum_exc(self):
        (PopRecorder2.get_instance(self.id))._sum_exc.clear()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder3_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder3.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder3.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder3.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).period_offset_ = val

    property I:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder3.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder3.get_instance(self.id)).v.clear()

    property w:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).w
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).w = val
    property record_w:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_w
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_w = val
    def clear_w(self):
        (PopRecorder3.get_instance(self.id)).w.clear()

    property g_exc:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).g_exc
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).g_exc = val
    property record_g_exc:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_g_exc
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_g_exc = val
    def clear_g_exc(self):
        (PopRecorder3.get_instance(self.id)).g_exc.clear()

    property g_inh:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder3.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder3.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder3.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder3.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder3.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder4_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder4.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder4.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder4.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).period_offset_ = val

    property I:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).I
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).I = val
    property record_I:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_I
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_I = val
    def clear_I(self):
        (PopRecorder4.get_instance(self.id)).I.clear()

    property v:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).v
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).v = val
    property record_v:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_v
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_v = val
    def clear_v(self):
        (PopRecorder4.get_instance(self.id)).v.clear()

    property w:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).w
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).w = val
    property record_w:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_w
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_w = val
    def clear_w(self):
        (PopRecorder4.get_instance(self.id)).w.clear()

    property g_exc:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).g_exc
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).g_exc = val
    property record_g_exc:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_g_exc
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_g_exc = val
    def clear_g_exc(self):
        (PopRecorder4.get_instance(self.id)).g_exc.clear()

    property g_inh:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).g_inh
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).g_inh = val
    property record_g_inh:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_g_inh
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_g_inh = val
    def clear_g_inh(self):
        (PopRecorder4.get_instance(self.id)).g_inh.clear()

    property r:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder4.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder4.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder4.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder4.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder5_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder5.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder5.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder5.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).period_offset_ = val

    property noise:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).noise
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).noise = val
    property record_noise:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_noise
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_noise = val
    def clear_noise(self):
        (PopRecorder5.get_instance(self.id)).noise.clear()

    property vm:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).vm
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).vm = val
    property record_vm:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_vm
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_vm = val
    def clear_vm(self):
        (PopRecorder5.get_instance(self.id)).vm.clear()

    property vmean:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).vmean
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).vmean = val
    property record_vmean:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_vmean
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_vmean = val
    def clear_vmean(self):
        (PopRecorder5.get_instance(self.id)).vmean.clear()

    property umeanLTD:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).umeanLTD
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).umeanLTD = val
    property record_umeanLTD:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_umeanLTD
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_umeanLTD = val
    def clear_umeanLTD(self):
        (PopRecorder5.get_instance(self.id)).umeanLTD.clear()

    property umeanLTP:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).umeanLTP
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).umeanLTP = val
    property record_umeanLTP:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_umeanLTP
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_umeanLTP = val
    def clear_umeanLTP(self):
        (PopRecorder5.get_instance(self.id)).umeanLTP.clear()

    property xtrace:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).xtrace
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).xtrace = val
    property record_xtrace:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_xtrace
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_xtrace = val
    def clear_xtrace(self):
        (PopRecorder5.get_instance(self.id)).xtrace.clear()

    property wad:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).wad
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).wad = val
    property record_wad:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_wad
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_wad = val
    def clear_wad(self):
        (PopRecorder5.get_instance(self.id)).wad.clear()

    property z:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).z
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).z = val
    property record_z:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_z
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_z = val
    def clear_z(self):
        (PopRecorder5.get_instance(self.id)).z.clear()

    property VT:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).VT
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).VT = val
    property record_VT:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_VT
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_VT = val
    def clear_VT(self):
        (PopRecorder5.get_instance(self.id)).VT.clear()

    property g_Exc:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).g_Exc
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).g_Exc = val
    property record_g_Exc:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_g_Exc
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_g_Exc = val
    def clear_g_Exc(self):
        (PopRecorder5.get_instance(self.id)).g_Exc.clear()

    property g_Inh:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).g_Inh
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).g_Inh = val
    property record_g_Inh:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_g_Inh
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_g_Inh = val
    def clear_g_Inh(self):
        (PopRecorder5.get_instance(self.id)).g_Inh.clear()

    property state:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).state
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).state = val
    property record_state:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_state
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_state = val
    def clear_state(self):
        (PopRecorder5.get_instance(self.id)).state.clear()

    property Spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).Spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).Spike = val
    property record_Spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_Spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_Spike = val
    def clear_Spike(self):
        (PopRecorder5.get_instance(self.id)).Spike.clear()

    property resetvar:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).resetvar
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).resetvar = val
    property record_resetvar:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_resetvar
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_resetvar = val
    def clear_resetvar(self):
        (PopRecorder5.get_instance(self.id)).resetvar.clear()

    property vmTemp:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).vmTemp
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).vmTemp = val
    property record_vmTemp:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_vmTemp
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_vmTemp = val
    def clear_vmTemp(self):
        (PopRecorder5.get_instance(self.id)).vmTemp.clear()

    property r:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder5.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder5.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder5.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder5.get_instance(self.id)).clear_spike()

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder6_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder6.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder6.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder6.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).period_offset_ = val

    property noise:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).noise
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).noise = val
    property record_noise:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_noise
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_noise = val
    def clear_noise(self):
        (PopRecorder6.get_instance(self.id)).noise.clear()

    property vm:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).vm
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).vm = val
    property record_vm:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_vm
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_vm = val
    def clear_vm(self):
        (PopRecorder6.get_instance(self.id)).vm.clear()

    property vmean:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).vmean
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).vmean = val
    property record_vmean:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_vmean
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_vmean = val
    def clear_vmean(self):
        (PopRecorder6.get_instance(self.id)).vmean.clear()

    property umeanLTD:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).umeanLTD
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).umeanLTD = val
    property record_umeanLTD:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_umeanLTD
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_umeanLTD = val
    def clear_umeanLTD(self):
        (PopRecorder6.get_instance(self.id)).umeanLTD.clear()

    property umeanLTP:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).umeanLTP
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).umeanLTP = val
    property record_umeanLTP:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_umeanLTP
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_umeanLTP = val
    def clear_umeanLTP(self):
        (PopRecorder6.get_instance(self.id)).umeanLTP.clear()

    property xtrace:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).xtrace
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).xtrace = val
    property record_xtrace:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_xtrace
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_xtrace = val
    def clear_xtrace(self):
        (PopRecorder6.get_instance(self.id)).xtrace.clear()

    property wad:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).wad
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).wad = val
    property record_wad:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_wad
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_wad = val
    def clear_wad(self):
        (PopRecorder6.get_instance(self.id)).wad.clear()

    property z:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).z
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).z = val
    property record_z:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_z
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_z = val
    def clear_z(self):
        (PopRecorder6.get_instance(self.id)).z.clear()

    property VT:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).VT
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).VT = val
    property record_VT:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_VT
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_VT = val
    def clear_VT(self):
        (PopRecorder6.get_instance(self.id)).VT.clear()

    property g_Exc:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).g_Exc
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).g_Exc = val
    property record_g_Exc:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_g_Exc
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_g_Exc = val
    def clear_g_Exc(self):
        (PopRecorder6.get_instance(self.id)).g_Exc.clear()

    property g_Inh:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).g_Inh
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).g_Inh = val
    property record_g_Inh:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_g_Inh
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_g_Inh = val
    def clear_g_Inh(self):
        (PopRecorder6.get_instance(self.id)).g_Inh.clear()

    property state:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).state
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).state = val
    property record_state:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_state
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_state = val
    def clear_state(self):
        (PopRecorder6.get_instance(self.id)).state.clear()

    property Spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).Spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).Spike = val
    property record_Spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_Spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_Spike = val
    def clear_Spike(self):
        (PopRecorder6.get_instance(self.id)).Spike.clear()

    property resetvar:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).resetvar
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).resetvar = val
    property record_resetvar:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_resetvar
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_resetvar = val
    def clear_resetvar(self):
        (PopRecorder6.get_instance(self.id)).resetvar.clear()

    property vmTemp:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).vmTemp
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).vmTemp = val
    property record_vmTemp:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_vmTemp
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_vmTemp = val
    def clear_vmTemp(self):
        (PopRecorder6.get_instance(self.id)).vmTemp.clear()

    property r:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder6.get_instance(self.id)).r.clear()

    property spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder6.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder6.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder6.get_instance(self.id)).clear_spike()

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder0.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder1.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder2_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder2.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder3_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder3.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder4_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder4.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder5_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder5.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder6_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder6.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder7_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder7.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder8_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder8.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder9_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder9.create_instance(ranks, period, period_offset, offset)

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder10_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder10.create_instance(ranks, period, period_offset, offset)


# User-defined functions


# User-defined constants


# Initialize/Destroy the network
def pyx_create():
    create_cpp_instances()
def pyx_initialize(double dt):
    initialize(dt)
def pyx_destroy():
    destroy_cpp_instances()

# Simple progressbar on the command line
def progress(count, total, status=''):
    """
    Prints a progress bar on the command line.

    adapted from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    Modification: The original code set the '\r' at the end, so the bar disappears when finished.
    I moved it to the front, so the last status remains.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

# Simulation for the given number of steps
def pyx_run(int nb_steps, progress_bar):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps % batch
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if nb > 1 and progress_bar:
                progress(i+1, nb, 'simulate()')
        if rest > 0:
            run(rest)

        if (progress_bar):
            print('\n')

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()


# Set number of threads
def set_number_threads(int n, core_list):
    setNumberThreads(n, core_list)


# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
