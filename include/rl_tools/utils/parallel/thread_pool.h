#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_PARALLEL_THREAD_POOL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_PARALLEL_THREAD_POOL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Fork-join pool for the elementwise, attention and optimizer loops of the BLAS CPU path. The GEMMs
// keep running on the BLAS library's own threads; everything between them used to run on the calling
// thread only, which was ~60% of a CPU training update (GELU tanh, attention, Adam).
//
// Use: create a pool once, then make it current for the calling thread with ScopedThreadPool while
// training. The CPU_BLAS operations look the pool up through current_thread_pool() and fall back to
// their serial loops when none is set (rollout inference, other threads, nested use).
//
// Workers spin for spin_us after a job (yielding to other threads after a short while, the GEMM
// threads of OpenBLAS spin the same way) and then sleep on a condition variable, so an idle pool costs
// nothing. Results do not depend on the number of threads: callers split their work into tasks whose
// boundaries depend only on the problem size, and every reduction is done per task in a fixed order.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::parallel{
    class ThreadPool{
    public:
        // n_threads counts the calling thread, so ThreadPool(1) runs everything serially
        explicit ThreadPool(unsigned n_threads, unsigned spin_us = 50): spin_us_(spin_us){
            for(unsigned i = 1; i < n_threads; i++){
                workers_.emplace_back([this]{ worker_loop(); });
            }
        }
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ~ThreadPool(){
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
                published_.store(~(unsigned long long)0, std::memory_order_release);
            }
            cv_work_.notify_all();
            for(auto& worker: workers_){
                worker.join();
            }
        }
        unsigned size() const{ return (unsigned)workers_.size() + 1; }
        // let utils::parallel::gemm split large gemms into single-threaded BLAS calls on this pool
        // (only correct if the BLAS library runs single-threaded, see blas.h)
        void set_tile_gemm(bool tile_gemm, unsigned max_tiles = 16){ tile_gemm_ = tile_gemm; gemm_max_tiles_ = max_tiles; }
        bool tile_gemm() const{ return tile_gemm_; }
        unsigned gemm_max_tiles() const{ return gemm_max_tiles_; }

        // Calls f(task_i) for task_i in [0, n_tasks) on the pool and returns when all calls returned.
        // A concurrent or nested call (the pool is busy) runs its tasks serially on the calling thread.
        template<typename F>
        void run(std::size_t n_tasks, F&& f){
            if(n_tasks == 0){
                return;
            }
            if(workers_.empty() || n_tasks == 1 || busy_.exchange(true, std::memory_order_acquire)){
                for(std::size_t task_i = 0; task_i < n_tasks; task_i++){
                    f(task_i);
                }
                return;
            }
            using FN = std::remove_reference_t<F>;
            Job job;
            job.fn = [](void* context, std::size_t task_i){ (*static_cast<FN*>(context))(task_i); };
            job.context = (void*)&f;
            job.n_tasks = n_tasks;
            bool wake = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                // workers still inside the previous job (they only find it exhausted) must leave
                // before its counters are reset
                cv_idle_.wait(lock, [this]{ return active_ == 0; });
                job_ = job;
                next_task_.store(0, std::memory_order_relaxed);
                done_tasks_.store(0, std::memory_order_relaxed);
                generation_++;
                published_.store(generation_, std::memory_order_release);
                wake = sleeping_ > 0;
            }
            if(wake){
                cv_work_.notify_all();
            }
            work(job);
            while(done_tasks_.load(std::memory_order_acquire) < n_tasks){
                pause();
            }
            busy_.store(false, std::memory_order_release);
        }
    private:
        struct Job{
            void (*fn)(void*, std::size_t) = nullptr;
            void* context = nullptr;
            std::size_t n_tasks = 0;
        };
        static void pause(){
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#else
            std::this_thread::yield();
#endif
        }
        void work(const Job& job){
            while(true){
                const std::size_t task_i = next_task_.fetch_add(1, std::memory_order_relaxed);
                if(task_i >= job.n_tasks){
                    break;
                }
                job.fn(job.context, task_i);
                done_tasks_.fetch_add(1, std::memory_order_release);
            }
        }
        void worker_loop(){
            unsigned long long seen = 0;
            while(true){
                // spin: pause first, then also yield the core so that spinning workers do not starve
                // the BLAS threads (or anything else) running between two jobs
                const auto spin_end = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us_);
                for(unsigned spin_i = 0; published_.load(std::memory_order_acquire) == seen; spin_i++){
                    if(spin_i < 256){
                        pause();
                    }
                    else{
                        if((spin_i & 15) == 0 && std::chrono::steady_clock::now() > spin_end){
                            break;
                        }
                        std::this_thread::yield();
                    }
                }
                Job job;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if(generation_ == seen && !stop_){
                        sleeping_++;
                        cv_work_.wait(lock, [&]{ return generation_ != seen || stop_; });
                        sleeping_--;
                    }
                    if(stop_){
                        return;
                    }
                    seen = generation_;
                    job = job_;
                    active_++;
                }
                work(job);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    active_--;
                    if(active_ == 0){
                        cv_idle_.notify_all();
                    }
                }
            }
        }
        std::vector<std::thread> workers_;
        const unsigned spin_us_;
        std::mutex mutex_;
        std::condition_variable cv_work_, cv_idle_;
        Job job_;
        unsigned long long generation_ = 0;
        std::atomic<unsigned long long> published_{0};
        std::atomic<std::size_t> next_task_{0}, done_tasks_{0};
        unsigned active_ = 0, sleeping_ = 0;
        bool stop_ = false;
        bool tile_gemm_ = false;
        unsigned gemm_max_tiles_ = 16;
        std::atomic<bool> busy_{false};
    };

    inline ThreadPool*& current_thread_pool_slot(){
        thread_local ThreadPool* pool = nullptr;
        return pool;
    }
    // the pool the CPU_BLAS operations of this thread may use (nullptr: serial)
    inline ThreadPool* current_thread_pool(){ return current_thread_pool_slot(); }

    class ScopedThreadPool{
    public:
        explicit ScopedThreadPool(ThreadPool* pool): previous_(current_thread_pool_slot()){ current_thread_pool_slot() = pool; }
        ~ScopedThreadPool(){ current_thread_pool_slot() = previous_; }
        ScopedThreadPool(const ScopedThreadPool&) = delete;
        ScopedThreadPool& operator=(const ScopedThreadPool&) = delete;
    private:
        ThreadPool* previous_;
    };

    // f(begin, end) for the chunks [i * grain, min((i + 1) * grain, n)) of [0, n), on the current pool if
    // there is one, else serially in order. The chunks are the same either way (they depend only on n
    // and grain), so callers can keep per-chunk partial results and get the same values with any
    // number of threads, or without a pool.
    template<typename TI, typename F>
    void parallel_for(TI n, TI grain, F&& f){
        if(n <= 0){
            return;
        }
        if(grain < 1){
            grain = 1;
        }
        const TI n_chunks = (n + grain - 1) / grain;
        auto chunk = [&](std::size_t chunk_i){
            const TI begin = (TI)chunk_i * grain;
            const TI end = n - begin > grain ? begin + grain : n;
            f(begin, end);
        };
        ThreadPool* pool = current_thread_pool();
        if(pool == nullptr || pool->size() == 1 || n_chunks <= 1){
            for(TI chunk_i = 0; chunk_i < n_chunks; chunk_i++){
                chunk((std::size_t)chunk_i);
            }
            return;
        }
        pool->run((std::size_t)n_chunks, chunk);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
