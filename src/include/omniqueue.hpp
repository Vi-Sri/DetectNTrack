#pragma once
#ifndef _SAFE_QUEUE_HPP_
#define _SAFE_QUEUE_HPP_

#include <queue>
#include <mutex>
#include <utility>
#include <condition_variable>

namespace omnispace {
    template <class T>
    class OmniQueue
    {
    public:
        OmniQueue(void)
            : q()
            , m()
            , c()
        {}

        ~OmniQueue(void){
            clear();
            std::cout<<"Clearing Queue"<<std::endl;
        }

        void enqueue(T t)
        {
            std::lock_guard<std::mutex> lock(m);
            q.push(t);
            c.notify_one();
        }

        T dequeue(void)
        {
            std::unique_lock<std::mutex> lock(m);
            while(q.empty())
            {
            c.wait(lock);
            }
            T val = q.front();
            q.pop();
            return val;
        }

        void clear(void)
            {
                std::queue<T> empty;
                std::swap( q, empty );
            }

        bool empty(void) 
            {
                return q.empty();
            }
        
        int size(void) 
            {
                return q.size();
            }

    private:
        std::queue<T> q;
        mutable std::mutex m;
        std::condition_variable c;
    };
}

#endif