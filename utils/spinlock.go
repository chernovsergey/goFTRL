package utils

import (
	"runtime"
	"sync"
	"sync/atomic"
)

type Locker struct {
	_    sync.Mutex // for copy protection compiler warning
	lock uintptr
}

func (l *Locker) Lock() {
	for !atomic.CompareAndSwapUintptr(&l.lock, 0, 1) {
		runtime.Gosched()
	}
}

func (l *Locker) Unlock() {
	atomic.StoreUintptr(&l.lock, 0)
}
