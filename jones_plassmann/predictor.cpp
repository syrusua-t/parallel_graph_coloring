#define THREADS_PER_BLOCK 256
#define INIT_HASH_CNT 4
#define MAX_HASH_CNT 32

enum strategy {
    CONSTANT,
    DOUBLE
};

class Predictor {
public:
    Predictor(strategy stg, int edge_cnt, int node_cnt) : stg_(stg), 
        edge_cnt_(edge_cnt), node_cnt_(node_cnt) {
        switch (stg_) {
            case CONSTANT:
                hash_cnt_ = INIT_HASH_CNT;
            default:
                break;
        }
    }
    
    int get_hash_cnt() {
        return hash_cnt_;
    }

    void update_hash_cnt(float hash_util) {
        switch (stg_) {
            case CONSTANT:
                break;
            default:
                break;
        }
    }

private:
    strategy stg_;
    int edge_cnt_;
    int node_cnt_;

    int hash_cnt_;
};