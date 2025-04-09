#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    template<typename T>
    using distributed = upcxx::dist_object<upcxx::global_ptr<T>>;
    upcxx::global_ptr<kmer_pair>* data;
    upcxx::global_ptr<int>* used;

    size_t my_size;

    size_t size() const noexcept { return size_g_; }

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer){
        uint64_t hash = kmer.hash();
        uint64_t probe = 0;
        bool success = false;
        do {
            uint64_t slot = (hash + probe++) % size();
            success = request_slot(slot);
            if (success) {
                write_slot(slot, kmer);
            }
        } while (!success && probe < size());
        return success;
    }

    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer){
        uint64_t hash = key_kmer.hash();
        uint64_t probe = 0;
        bool success = false;
        do {
            uint64_t slot = (hash + probe++) % size();
            if (slot_used(slot)) {
                val_kmer = read_slot(slot);
                if (val_kmer.kmer == key_kmer) {
                    success = true;
                }
            }
        } while (!success && probe < size());
        return success;
    }

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer){
        auto slot_ptr = data[slot/stride_]+slot%stride_;
        upcxx::rput(kmer, slot_ptr).wait();
    }

    kmer_pair read_slot(uint64_t slot){
        auto slot_ptr = data[slot/stride_]+slot%stride_;
        return upcxx::rget(slot_ptr).wait();    
    }

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot){
        auto slot_ptr = used[slot/stride_]+slot%stride_;
        int slot_val = ad_.fetch_add(slot_ptr, 1, std::memory_order_relaxed).wait();
        return (slot_val == 0);
    }

    bool slot_used(uint64_t slot){
        auto slot_ptr = used[slot/stride_]+slot%stride_;
        return upcxx::rget(slot_ptr).wait() != 0;
    }

private:
    int N_g_, n_;
    size_t stride_, offset_, size_g_;
    distributed<kmer_pair>* data_g_;
    distributed<int>* used_g_;
    upcxx::atomic_domain<int> ad_;
};

HashMap::HashMap(size_t size)
    : ad_({upcxx::atomic_op::fetch_add}),
    size_g_(size)
{
    N_g_ = upcxx::rank_n();
    n_ = upcxx::rank_me();
    stride_ = (size+N_g_-1)/N_g_;
    offset_ = n_*stride_;
    my_size = std::min(stride_, size-offset_);
    data_g_ = new distributed<kmer_pair>(upcxx::new_array<kmer_pair>(my_size));
    used_g_ = new distributed<int>(upcxx::new_array<int>(my_size));
    data = new upcxx::global_ptr<kmer_pair>[N_g_];
    used = new upcxx::global_ptr<int>[N_g_];
    for (int i = 0; i < N_g_; ++i) {
        data[i] = data_g_->fetch(i).wait();
        used[i] = used_g_->fetch(i).wait();
    }
}

// bool HashMap::insert(const kmer_pair& kmer) {
//     uint64_t hash = kmer.hash();
//     uint64_t probe = 0;
//     bool success = false;
//     do {
//         uint64_t slot = (hash + probe) % size();
//         success = request_slot(slot);
//         if (success) {
//             write_slot(slot, kmer);
//         }
//         ++probe;
//     } while (!success && probe < size());
//     return success;
// }

// bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
//     uint64_t hash = key_kmer.hash();
//     uint64_t probe = 0;
//     bool success = false;
//     do {
//         uint64_t slot = (hash + probe) % size();
//         if (slot_used(slot)) {
//             val_kmer = read_slot(slot);
//             if (val_kmer.kmer == key_kmer) {
//                 success = true;
//             }
//         }
//         ++probe;
//     } while (!success && probe < size());
//     return success;
// }

// bool HashMap::slot_used(uint64_t slot) {
//     auto slot_ptr = used[slot/stride_]+slot%stride_;
//     return upcxx::rget(slot_ptr).wait() != 0;
// }

// void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
//     auto slot_ptr = data[slot/stride_]+slot%stride_;
//     upcxx::rput(kmer, slot_ptr).wait();
// }

// kmer_pair HashMap::read_slot(uint64_t slot) {
//     auto slot_ptr = data[slot/stride_]+slot%stride_;
//     return upcxx::rget(slot_ptr).wait();
// }

// bool HashMap::request_slot(uint64_t slot) {
//     auto slot_ptr = used[slot/stride_]+slot%stride_;
//     int slot_val = ad_.fetch_add(slot_ptr, 1, std::memory_order_relaxed).wait();
//     return (slot_val == 0);
// }

// size_t HashMap::size() const noexcept { return size_g_; }
