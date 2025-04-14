#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
using namespace std;

struct HashMap {
    template<typename T>
    using distributed = upcxx::dist_object<upcxx::global_ptr<T>>;
    upcxx::global_ptr<kmer_pair>* data;
    upcxx::global_ptr<int>* used;

    size_t my_size;

    size_t size() const noexcept { return global_size; }

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

        auto slot_pointer = data[slot / stride] + (slot%stride);
        upcxx::rput(kmer, slot_pointer).wait();
    }

    kmer_pair read_slot(uint64_t slot){

        auto slot_pointer = data[slot / stride] + (slot%stride);
        return upcxx::rget(slot_pointer).wait();    
    }

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot){
        int slot_val;

        auto slot_pointer = used[slot / stride] + (slot%stride);
        slot_val = atomic_flags.fetch_add(slot_pointer, 1, memory_order_relaxed).wait();
        return (slot_val == 0);
    }

    bool slot_used(uint64_t slot){

        auto slot_pointer = used[slot / stride] + (slot%stride);
        return upcxx::rget(slot_pointer).wait() != 0;
    }

private:
    int world_size, rank;
    size_t stride, offset, global_size;
    distributed<kmer_pair>* data_g_;
    distributed<int>* slot_flag_ptr;
    upcxx::atomic_domain<int> atomic_flags;
};

HashMap::HashMap(size_t size)
    : atomic_flags({upcxx::atomic_op::fetch_add}),
    global_size(size)
{
    world_size = upcxx::rank_n();
    rank = upcxx::rank_me();

    stride = (size+ world_size - 1) / world_size;
    offset = rank * stride;
    my_size = min(stride, size-offset);

    data_g_ = new distributed<kmer_pair>(upcxx::new_array<kmer_pair>(my_size));
    slot_flag_ptr = new distributed<int>(upcxx::new_array<int>(my_size));
    
    data = new upcxx::global_ptr<kmer_pair>[world_size];
    used = new upcxx::global_ptr<int>[world_size];

    int i;
    for (i = 0; i < world_size; i++) {
        data[i] = data_g_->fetch(i).wait();
        used[i] = slot_flag_ptr->fetch(i).wait();
    }
}
