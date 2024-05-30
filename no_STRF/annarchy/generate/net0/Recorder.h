#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

    }

public:
    ~PopRecorder0() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::~PopRecorder0() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder0* get_instance(int id) {
        return static_cast<PopRecorder0*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder0::clear() - this = " << this << std::endl;
    #endif


        removeRecorder(this);
    }



};

class PopRecorder1 : public Monitor
{
protected:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") instantiated." << std::endl;
    #endif

        this->_sum_exc = std::vector< std::vector< double > >();
        this->record__sum_exc = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:
    ~PopRecorder1() {
    #ifdef _DEBUG
        std::cout << "PopRecorder1::~PopRecorder1() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder1* get_instance(int id) {
        return static_cast<PopRecorder1*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder1::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop1.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->_sum_exc.push_back(pop1._sum_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1._sum_exc[this->ranks[i]]);
                }
                this->_sum_exc.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder1::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->_sum_exc.begin(); it != this->_sum_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->_sum_exc.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    

        removeRecorder(this);
    }



    // Local variable _sum_exc
    std::vector< std::vector< double > > _sum_exc ;
    bool record__sum_exc ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder2 : public Monitor
{
protected:
    PopRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << this << ") instantiated." << std::endl;
    #endif

        this->_sum_exc = std::vector< std::vector< double > >();
        this->record__sum_exc = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
    }

public:
    ~PopRecorder2() {
    #ifdef _DEBUG
        std::cout << "PopRecorder2::~PopRecorder2() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder2* get_instance(int id) {
        return static_cast<PopRecorder2*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder2::record()" << std::endl;
    #endif

        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop2.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->_sum_exc.push_back(pop2._sum_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2._sum_exc[this->ranks[i]]);
                }
                this->_sum_exc.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder2::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->_sum_exc.begin(); it != this->_sum_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->_sum_exc.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    

        removeRecorder(this);
    }



    // Local variable _sum_exc
    std::vector< std::vector< double > > _sum_exc ;
    bool record__sum_exc ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
};

class PopRecorder3 : public Monitor
{
protected:
    PopRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_exc = std::vector< std::vector< double > >();
        this->record_g_exc = false; 
        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->w = std::vector< std::vector< double > >();
        this->record_w = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop3.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false;

    }

public:
    ~PopRecorder3() {
    #ifdef _DEBUG
        std::cout << "PopRecorder3::~PopRecorder3() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder3* get_instance(int id) {
        return static_cast<PopRecorder3*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder3::record()" << std::endl;
    #endif

        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop3.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop3.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->w.push_back(pop3.w);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.w[this->ranks[i]]);
                }
                this->w.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop3.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop3.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop3.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop3.spiked[i])!=this->ranks.end() ){
                        this->spike[pop3.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_exc.push_back(pop3.g_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.g_exc[this->ranks[i]]);
                }
                this->g_exc.push_back(tmp);
            }
        }
        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_inh.push_back(pop3.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable w
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it=w.begin(); it!= w.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder3::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->g_exc.begin(); it != this->g_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_exc.clear();
    
        for(auto it = this->g_inh.begin(); it != this->g_inh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_inh.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->w.begin(); it != this->w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->w.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        

        removeRecorder(this);
    }



    // Local variable g_exc
    std::vector< std::vector< double > > g_exc ;
    bool record_g_exc ; 
    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable w
    std::vector< std::vector< double > > w ;
    bool record_w ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //spike.clear();
    }

};

class PopRecorder4 : public Monitor
{
protected:
    PopRecorder4(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_exc = std::vector< std::vector< double > >();
        this->record_g_exc = false; 
        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->I = std::vector< std::vector< double > >();
        this->record_I = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->w = std::vector< std::vector< double > >();
        this->record_w = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop4.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false;

    }

public:
    ~PopRecorder4() {
    #ifdef _DEBUG
        std::cout << "PopRecorder4::~PopRecorder4() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder4(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder4 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder4* get_instance(int id) {
        return static_cast<PopRecorder4*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder4::record()" << std::endl;
    #endif

        if(this->record_I && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->I.push_back(pop4.I);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.I[this->ranks[i]]);
                }
                this->I.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop4.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->w.push_back(pop4.w);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.w[this->ranks[i]]);
                }
                this->w.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop4.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop4.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop4.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop4.spiked[i])!=this->ranks.end() ){
                        this->spike[pop4.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_exc.push_back(pop4.g_exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.g_exc[this->ranks[i]]);
                }
                this->g_exc.push_back(tmp);
            }
        }
        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_inh.push_back(pop4.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop4.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable I
        size_in_bytes += sizeof(std::vector<double>) * I.capacity();
        for(auto it=I.begin(); it!= I.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable w
        size_in_bytes += sizeof(std::vector<double>) * w.capacity();
        for(auto it=w.begin(); it!= w.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder4::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->g_exc.begin(); it != this->g_exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_exc.clear();
    
        for(auto it = this->g_inh.begin(); it != this->g_inh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_inh.clear();
    
        for(auto it = this->I.begin(); it != this->I.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->I.clear();
    
        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->w.begin(); it != this->w.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->w.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        

        removeRecorder(this);
    }



    // Local variable g_exc
    std::vector< std::vector< double > > g_exc ;
    bool record_g_exc ; 
    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable I
    std::vector< std::vector< double > > I ;
    bool record_I ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable w
    std::vector< std::vector< double > > w ;
    bool record_w ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //spike.clear();
    }

};

class PopRecorder5 : public Monitor
{
protected:
    PopRecorder5(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_Exc = std::vector< std::vector< double > >();
        this->record_g_Exc = false; 
        this->g_Inh = std::vector< std::vector< double > >();
        this->record_g_Inh = false; 
        this->noise = std::vector< std::vector< double > >();
        this->record_noise = false; 
        this->vm = std::vector< std::vector< double > >();
        this->record_vm = false; 
        this->vmean = std::vector< std::vector< double > >();
        this->record_vmean = false; 
        this->umeanLTD = std::vector< std::vector< double > >();
        this->record_umeanLTD = false; 
        this->umeanLTP = std::vector< std::vector< double > >();
        this->record_umeanLTP = false; 
        this->xtrace = std::vector< std::vector< double > >();
        this->record_xtrace = false; 
        this->wad = std::vector< std::vector< double > >();
        this->record_wad = false; 
        this->z = std::vector< std::vector< double > >();
        this->record_z = false; 
        this->VT = std::vector< std::vector< double > >();
        this->record_VT = false; 
        this->state = std::vector< std::vector< double > >();
        this->record_state = false; 
        this->Spike = std::vector< std::vector< double > >();
        this->record_Spike = false; 
        this->resetvar = std::vector< std::vector< double > >();
        this->record_resetvar = false; 
        this->vmTemp = std::vector< std::vector< double > >();
        this->record_vmTemp = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop5.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false;

    }

public:
    ~PopRecorder5() {
    #ifdef _DEBUG
        std::cout << "PopRecorder5::~PopRecorder5() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder5(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder5 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder5* get_instance(int id) {
        return static_cast<PopRecorder5*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder5::record()" << std::endl;
    #endif

        if(this->record_noise && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->noise.push_back(pop5.noise);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.noise[this->ranks[i]]);
                }
                this->noise.push_back(tmp);
            }
        }
        if(this->record_vm && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vm.push_back(pop5.vm);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.vm[this->ranks[i]]);
                }
                this->vm.push_back(tmp);
            }
        }
        if(this->record_vmean && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vmean.push_back(pop5.vmean);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.vmean[this->ranks[i]]);
                }
                this->vmean.push_back(tmp);
            }
        }
        if(this->record_umeanLTD && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->umeanLTD.push_back(pop5.umeanLTD);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.umeanLTD[this->ranks[i]]);
                }
                this->umeanLTD.push_back(tmp);
            }
        }
        if(this->record_umeanLTP && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->umeanLTP.push_back(pop5.umeanLTP);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.umeanLTP[this->ranks[i]]);
                }
                this->umeanLTP.push_back(tmp);
            }
        }
        if(this->record_xtrace && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->xtrace.push_back(pop5.xtrace);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.xtrace[this->ranks[i]]);
                }
                this->xtrace.push_back(tmp);
            }
        }
        if(this->record_wad && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->wad.push_back(pop5.wad);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.wad[this->ranks[i]]);
                }
                this->wad.push_back(tmp);
            }
        }
        if(this->record_z && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->z.push_back(pop5.z);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.z[this->ranks[i]]);
                }
                this->z.push_back(tmp);
            }
        }
        if(this->record_VT && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->VT.push_back(pop5.VT);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.VT[this->ranks[i]]);
                }
                this->VT.push_back(tmp);
            }
        }
        if(this->record_state && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->state.push_back(pop5.state);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.state[this->ranks[i]]);
                }
                this->state.push_back(tmp);
            }
        }
        if(this->record_Spike && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->Spike.push_back(pop5.Spike);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.Spike[this->ranks[i]]);
                }
                this->Spike.push_back(tmp);
            }
        }
        if(this->record_resetvar && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->resetvar.push_back(pop5.resetvar);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.resetvar[this->ranks[i]]);
                }
                this->resetvar.push_back(tmp);
            }
        }
        if(this->record_vmTemp && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vmTemp.push_back(pop5.vmTemp);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.vmTemp[this->ranks[i]]);
                }
                this->vmTemp.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop5.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop5.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop5.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop5.spiked[i])!=this->ranks.end() ){
                        this->spike[pop5.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_Exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_Exc.push_back(pop5.g_Exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.g_Exc[this->ranks[i]]);
                }
                this->g_Exc.push_back(tmp);
            }
        }
        if(this->record_g_Inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_Inh.push_back(pop5.g_Inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop5.g_Inh[this->ranks[i]]);
                }
                this->g_Inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable noise
        size_in_bytes += sizeof(std::vector<double>) * noise.capacity();
        for(auto it=noise.begin(); it!= noise.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vm
        size_in_bytes += sizeof(std::vector<double>) * vm.capacity();
        for(auto it=vm.begin(); it!= vm.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vmean
        size_in_bytes += sizeof(std::vector<double>) * vmean.capacity();
        for(auto it=vmean.begin(); it!= vmean.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable umeanLTD
        size_in_bytes += sizeof(std::vector<double>) * umeanLTD.capacity();
        for(auto it=umeanLTD.begin(); it!= umeanLTD.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable umeanLTP
        size_in_bytes += sizeof(std::vector<double>) * umeanLTP.capacity();
        for(auto it=umeanLTP.begin(); it!= umeanLTP.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable xtrace
        size_in_bytes += sizeof(std::vector<double>) * xtrace.capacity();
        for(auto it=xtrace.begin(); it!= xtrace.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable wad
        size_in_bytes += sizeof(std::vector<double>) * wad.capacity();
        for(auto it=wad.begin(); it!= wad.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable z
        size_in_bytes += sizeof(std::vector<double>) * z.capacity();
        for(auto it=z.begin(); it!= z.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable VT
        size_in_bytes += sizeof(std::vector<double>) * VT.capacity();
        for(auto it=VT.begin(); it!= VT.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable state
        size_in_bytes += sizeof(std::vector<double>) * state.capacity();
        for(auto it=state.begin(); it!= state.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable Spike
        size_in_bytes += sizeof(std::vector<double>) * Spike.capacity();
        for(auto it=Spike.begin(); it!= Spike.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable resetvar
        size_in_bytes += sizeof(std::vector<double>) * resetvar.capacity();
        for(auto it=resetvar.begin(); it!= resetvar.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vmTemp
        size_in_bytes += sizeof(std::vector<double>) * vmTemp.capacity();
        for(auto it=vmTemp.begin(); it!= vmTemp.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder5::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->g_Exc.begin(); it != this->g_Exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_Exc.clear();
    
        for(auto it = this->g_Inh.begin(); it != this->g_Inh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_Inh.clear();
    
        for(auto it = this->noise.begin(); it != this->noise.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->noise.clear();
    
        for(auto it = this->vm.begin(); it != this->vm.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vm.clear();
    
        for(auto it = this->vmean.begin(); it != this->vmean.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vmean.clear();
    
        for(auto it = this->umeanLTD.begin(); it != this->umeanLTD.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->umeanLTD.clear();
    
        for(auto it = this->umeanLTP.begin(); it != this->umeanLTP.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->umeanLTP.clear();
    
        for(auto it = this->xtrace.begin(); it != this->xtrace.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->xtrace.clear();
    
        for(auto it = this->wad.begin(); it != this->wad.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->wad.clear();
    
        for(auto it = this->z.begin(); it != this->z.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->z.clear();
    
        for(auto it = this->VT.begin(); it != this->VT.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->VT.clear();
    
        for(auto it = this->state.begin(); it != this->state.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->state.clear();
    
        for(auto it = this->Spike.begin(); it != this->Spike.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->Spike.clear();
    
        for(auto it = this->resetvar.begin(); it != this->resetvar.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->resetvar.clear();
    
        for(auto it = this->vmTemp.begin(); it != this->vmTemp.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vmTemp.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        

        removeRecorder(this);
    }



    // Local variable g_Exc
    std::vector< std::vector< double > > g_Exc ;
    bool record_g_Exc ; 
    // Local variable g_Inh
    std::vector< std::vector< double > > g_Inh ;
    bool record_g_Inh ; 
    // Local variable noise
    std::vector< std::vector< double > > noise ;
    bool record_noise ; 
    // Local variable vm
    std::vector< std::vector< double > > vm ;
    bool record_vm ; 
    // Local variable vmean
    std::vector< std::vector< double > > vmean ;
    bool record_vmean ; 
    // Local variable umeanLTD
    std::vector< std::vector< double > > umeanLTD ;
    bool record_umeanLTD ; 
    // Local variable umeanLTP
    std::vector< std::vector< double > > umeanLTP ;
    bool record_umeanLTP ; 
    // Local variable xtrace
    std::vector< std::vector< double > > xtrace ;
    bool record_xtrace ; 
    // Local variable wad
    std::vector< std::vector< double > > wad ;
    bool record_wad ; 
    // Local variable z
    std::vector< std::vector< double > > z ;
    bool record_z ; 
    // Local variable VT
    std::vector< std::vector< double > > VT ;
    bool record_VT ; 
    // Local variable state
    std::vector< std::vector< double > > state ;
    bool record_state ; 
    // Local variable Spike
    std::vector< std::vector< double > > Spike ;
    bool record_Spike ; 
    // Local variable resetvar
    std::vector< std::vector< double > > resetvar ;
    bool record_resetvar ; 
    // Local variable vmTemp
    std::vector< std::vector< double > > vmTemp ;
    bool record_vmTemp ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //spike.clear();
    }

};

class PopRecorder6 : public Monitor
{
protected:
    PopRecorder6(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder6 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_Exc = std::vector< std::vector< double > >();
        this->record_g_Exc = false; 
        this->g_Inh = std::vector< std::vector< double > >();
        this->record_g_Inh = false; 
        this->noise = std::vector< std::vector< double > >();
        this->record_noise = false; 
        this->vm = std::vector< std::vector< double > >();
        this->record_vm = false; 
        this->vmean = std::vector< std::vector< double > >();
        this->record_vmean = false; 
        this->umeanLTD = std::vector< std::vector< double > >();
        this->record_umeanLTD = false; 
        this->umeanLTP = std::vector< std::vector< double > >();
        this->record_umeanLTP = false; 
        this->xtrace = std::vector< std::vector< double > >();
        this->record_xtrace = false; 
        this->wad = std::vector< std::vector< double > >();
        this->record_wad = false; 
        this->z = std::vector< std::vector< double > >();
        this->record_z = false; 
        this->VT = std::vector< std::vector< double > >();
        this->record_VT = false; 
        this->state = std::vector< std::vector< double > >();
        this->record_state = false; 
        this->Spike = std::vector< std::vector< double > >();
        this->record_Spike = false; 
        this->resetvar = std::vector< std::vector< double > >();
        this->record_resetvar = false; 
        this->vmTemp = std::vector< std::vector< double > >();
        this->record_vmTemp = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop6.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false;

    }

public:
    ~PopRecorder6() {
    #ifdef _DEBUG
        std::cout << "PopRecorder6::~PopRecorder6() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder6(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder6 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder6* get_instance(int id) {
        return static_cast<PopRecorder6*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder6::record()" << std::endl;
    #endif

        if(this->record_noise && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->noise.push_back(pop6.noise);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.noise[this->ranks[i]]);
                }
                this->noise.push_back(tmp);
            }
        }
        if(this->record_vm && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vm.push_back(pop6.vm);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.vm[this->ranks[i]]);
                }
                this->vm.push_back(tmp);
            }
        }
        if(this->record_vmean && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vmean.push_back(pop6.vmean);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.vmean[this->ranks[i]]);
                }
                this->vmean.push_back(tmp);
            }
        }
        if(this->record_umeanLTD && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->umeanLTD.push_back(pop6.umeanLTD);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.umeanLTD[this->ranks[i]]);
                }
                this->umeanLTD.push_back(tmp);
            }
        }
        if(this->record_umeanLTP && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->umeanLTP.push_back(pop6.umeanLTP);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.umeanLTP[this->ranks[i]]);
                }
                this->umeanLTP.push_back(tmp);
            }
        }
        if(this->record_xtrace && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->xtrace.push_back(pop6.xtrace);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.xtrace[this->ranks[i]]);
                }
                this->xtrace.push_back(tmp);
            }
        }
        if(this->record_wad && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->wad.push_back(pop6.wad);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.wad[this->ranks[i]]);
                }
                this->wad.push_back(tmp);
            }
        }
        if(this->record_z && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->z.push_back(pop6.z);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.z[this->ranks[i]]);
                }
                this->z.push_back(tmp);
            }
        }
        if(this->record_VT && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->VT.push_back(pop6.VT);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.VT[this->ranks[i]]);
                }
                this->VT.push_back(tmp);
            }
        }
        if(this->record_state && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->state.push_back(pop6.state);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.state[this->ranks[i]]);
                }
                this->state.push_back(tmp);
            }
        }
        if(this->record_Spike && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->Spike.push_back(pop6.Spike);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.Spike[this->ranks[i]]);
                }
                this->Spike.push_back(tmp);
            }
        }
        if(this->record_resetvar && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->resetvar.push_back(pop6.resetvar);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.resetvar[this->ranks[i]]);
                }
                this->resetvar.push_back(tmp);
            }
        }
        if(this->record_vmTemp && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vmTemp.push_back(pop6.vmTemp);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.vmTemp[this->ranks[i]]);
                }
                this->vmTemp.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop6.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop6.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop6.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop6.spiked[i])!=this->ranks.end() ){
                        this->spike[pop6.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_Exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_Exc.push_back(pop6.g_Exc);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.g_Exc[this->ranks[i]]);
                }
                this->g_Exc.push_back(tmp);
            }
        }
        if(this->record_g_Inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_Inh.push_back(pop6.g_Inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop6.g_Inh[this->ranks[i]]);
                }
                this->g_Inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable noise
        size_in_bytes += sizeof(std::vector<double>) * noise.capacity();
        for(auto it=noise.begin(); it!= noise.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vm
        size_in_bytes += sizeof(std::vector<double>) * vm.capacity();
        for(auto it=vm.begin(); it!= vm.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vmean
        size_in_bytes += sizeof(std::vector<double>) * vmean.capacity();
        for(auto it=vmean.begin(); it!= vmean.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable umeanLTD
        size_in_bytes += sizeof(std::vector<double>) * umeanLTD.capacity();
        for(auto it=umeanLTD.begin(); it!= umeanLTD.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable umeanLTP
        size_in_bytes += sizeof(std::vector<double>) * umeanLTP.capacity();
        for(auto it=umeanLTP.begin(); it!= umeanLTP.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable xtrace
        size_in_bytes += sizeof(std::vector<double>) * xtrace.capacity();
        for(auto it=xtrace.begin(); it!= xtrace.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable wad
        size_in_bytes += sizeof(std::vector<double>) * wad.capacity();
        for(auto it=wad.begin(); it!= wad.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable z
        size_in_bytes += sizeof(std::vector<double>) * z.capacity();
        for(auto it=z.begin(); it!= z.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable VT
        size_in_bytes += sizeof(std::vector<double>) * VT.capacity();
        for(auto it=VT.begin(); it!= VT.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable state
        size_in_bytes += sizeof(std::vector<double>) * state.capacity();
        for(auto it=state.begin(); it!= state.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable Spike
        size_in_bytes += sizeof(std::vector<double>) * Spike.capacity();
        for(auto it=Spike.begin(); it!= Spike.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable resetvar
        size_in_bytes += sizeof(std::vector<double>) * resetvar.capacity();
        for(auto it=resetvar.begin(); it!= resetvar.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable vmTemp
        size_in_bytes += sizeof(std::vector<double>) * vmTemp.capacity();
        for(auto it=vmTemp.begin(); it!= vmTemp.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        // record spike events
        size_in_bytes += sizeof(spike);
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            size_in_bytes += sizeof(int); // key
            size_in_bytes += sizeof(long int) * (it->second).capacity(); // value
        }
                
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder6::clear() - this = " << this << std::endl;
    #endif

        for(auto it = this->g_Exc.begin(); it != this->g_Exc.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_Exc.clear();
    
        for(auto it = this->g_Inh.begin(); it != this->g_Inh.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->g_Inh.clear();
    
        for(auto it = this->noise.begin(); it != this->noise.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->noise.clear();
    
        for(auto it = this->vm.begin(); it != this->vm.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vm.clear();
    
        for(auto it = this->vmean.begin(); it != this->vmean.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vmean.clear();
    
        for(auto it = this->umeanLTD.begin(); it != this->umeanLTD.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->umeanLTD.clear();
    
        for(auto it = this->umeanLTP.begin(); it != this->umeanLTP.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->umeanLTP.clear();
    
        for(auto it = this->xtrace.begin(); it != this->xtrace.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->xtrace.clear();
    
        for(auto it = this->wad.begin(); it != this->wad.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->wad.clear();
    
        for(auto it = this->z.begin(); it != this->z.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->z.clear();
    
        for(auto it = this->VT.begin(); it != this->VT.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->VT.clear();
    
        for(auto it = this->state.begin(); it != this->state.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->state.clear();
    
        for(auto it = this->Spike.begin(); it != this->Spike.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->Spike.clear();
    
        for(auto it = this->resetvar.begin(); it != this->resetvar.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->resetvar.clear();
    
        for(auto it = this->vmTemp.begin(); it != this->vmTemp.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->vmTemp.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        

        removeRecorder(this);
    }



    // Local variable g_Exc
    std::vector< std::vector< double > > g_Exc ;
    bool record_g_Exc ; 
    // Local variable g_Inh
    std::vector< std::vector< double > > g_Inh ;
    bool record_g_Inh ; 
    // Local variable noise
    std::vector< std::vector< double > > noise ;
    bool record_noise ; 
    // Local variable vm
    std::vector< std::vector< double > > vm ;
    bool record_vm ; 
    // Local variable vmean
    std::vector< std::vector< double > > vmean ;
    bool record_vmean ; 
    // Local variable umeanLTD
    std::vector< std::vector< double > > umeanLTD ;
    bool record_umeanLTD ; 
    // Local variable umeanLTP
    std::vector< std::vector< double > > umeanLTP ;
    bool record_umeanLTP ; 
    // Local variable xtrace
    std::vector< std::vector< double > > xtrace ;
    bool record_xtrace ; 
    // Local variable wad
    std::vector< std::vector< double > > wad ;
    bool record_wad ; 
    // Local variable z
    std::vector< std::vector< double > > z ;
    bool record_z ; 
    // Local variable VT
    std::vector< std::vector< double > > VT ;
    bool record_VT ; 
    // Local variable state
    std::vector< std::vector< double > > state ;
    bool record_state ; 
    // Local variable Spike
    std::vector< std::vector< double > > Spike ;
    bool record_Spike ; 
    // Local variable resetvar
    std::vector< std::vector< double > > resetvar ;
    bool record_resetvar ; 
    // Local variable vmTemp
    std::vector< std::vector< double > > vmTemp ;
    bool record_vmTemp ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //spike.clear();
    }

};

class ProjRecorder0 : public Monitor
{
protected:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder0* get_instance(int id) {
        return static_cast<ProjRecorder0*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor0::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder1 : public Monitor
{
protected:
    ProjRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj1.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder1* get_instance(int id) {
        return static_cast<ProjRecorder1*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor1::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder2 : public Monitor
{
protected:
    ProjRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder2 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj2.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder2* get_instance(int id) {
        return static_cast<ProjRecorder2*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor2::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder3 : public Monitor
{
protected:
    ProjRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder3 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj3.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder3* get_instance(int id) {
        return static_cast<ProjRecorder3*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor3::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder4 : public Monitor
{
protected:
    ProjRecorder4(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder4 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj4.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder4(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder4 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder4* get_instance(int id) {
        return static_cast<ProjRecorder4*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor4::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder5 : public Monitor
{
protected:
    ProjRecorder5(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder5 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj5.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder5(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder5 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder5* get_instance(int id) {
        return static_cast<ProjRecorder5*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor5::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder6 : public Monitor
{
protected:
    ProjRecorder6(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder6 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj6.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder6(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder6 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder6* get_instance(int id) {
        return static_cast<ProjRecorder6*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor6::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder7 : public Monitor
{
protected:
    ProjRecorder7(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder7 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj7.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder7(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder7 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder7* get_instance(int id) {
        return static_cast<ProjRecorder7*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor7::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder8 : public Monitor
{
protected:
    ProjRecorder8(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder8 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj8.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder8(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder8 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder8* get_instance(int id) {
        return static_cast<ProjRecorder8*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor8::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder9 : public Monitor
{
protected:
    ProjRecorder9(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder9 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj9.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder9(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder9 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder9* get_instance(int id) {
        return static_cast<ProjRecorder9*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor9::clear()." << std::endl;
    #endif

    }


};

class ProjRecorder10 : public Monitor
{
protected:
    ProjRecorder10(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder10 (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj10.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();


    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder10(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder10 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder10* get_instance(int id) {
        return static_cast<ProjRecorder10*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;



        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor10::clear()." << std::endl;
    #endif

    }


};

