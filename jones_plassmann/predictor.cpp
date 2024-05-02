#define THREADS_PER_BLOCK 256
#define INIT_HASH_CNT 4
#define MAX_HASH_CNT 4096

enum strategy {
    CONSTANT,
    DOUBLE
};

class Predictor {
public:
    Predictor(strategy stg, int edge_cnt, int node_cnt, bool verbose = false) : stg_(stg), 
        edge_cnt_(edge_cnt), node_cnt_(node_cnt), verbose_(verbose) {
        switch (stg_) {
            default:
                hash_cnt_ = INIT_HASH_CNT;
                break;
        }
    }
    
    int get_hash_cnt() {
        return hash_cnt_;
    }

    void update_hash_cnt(float hash_util) {
        int hash_cnt = hash_cnt_;
        switch (stg_) {
            case CONSTANT:
                break;
            case DOUBLE:
                if (hash_util < util_limit_double && hash_cnt_ * 2 <= MAX_HASH_CNT) {
                    hash_cnt_ *= 2;
                }
                break;
            default:
                break;
        }
        if (verbose_) {
            printf("Iteration[%d]: hash utilization = %f, hash_cnt: %d -> %d\n", it_++, hash_util, hash_cnt, hash_cnt_);
        }
    }

private:
    strategy stg_;
    int edge_cnt_;
    int node_cnt_;
    int it_ = 0;
    bool verbose_;
    int hash_cnt_;

    const float util_limit_double = 0.3f;
};