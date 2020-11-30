#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#define MAX_SEQ_NUMBER 	(12)
#define FRAME_NUM (3)

int max_seq_number = 0;
#define MAX_THREAD_NUMBER (6)

#define MAX_PRINT_NUM	(10)

struct Slice {
	int seq;
//	struct Slice *next;
	int end;
};

struct thread_param {
//	int seq;
	int thread_no;
	pthread_mutex_t  my_thread_mutex;
//	pthread_cond_t  my_thread_cond;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
//	struct Slice *top;
};



void wait_write_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, &param->write_bitstream_next_mutex);
	int ret = pthread_mutex_lock(&param->write_bitstream_my_mutex);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	return;
}
void start_write_next_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, param->write_bitstream_next_mutex);
	if (param->write_bitstream_next_mutex != NULL ) {
		int ret = pthread_mutex_unlock(param->write_bitstream_next_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return;
		}
	}
	return;
}

pthread_mutex_t end_frame_mutex;

void *thread_start_routin(void *arg)
{
	struct thread_param *param =  (struct thread_param*)arg;

	char fifoname[1024];
	sprintf(fifoname, "/tmp/fifo%d", param->thread_no);	
	mkfifo(fifoname, 0666);

	int fd = open(fifoname, O_RDONLY);

	
	for(;;) {
		struct Slice slice;
		read(fd, &slice, sizeof(slice));

		int counter = 0;
			int i;
			int num = rand();
			for (i = 0; i< num % MAX_PRINT_NUM;i++) {
				printf("dct seq=%d max=%d counter=%d\n", slice.seq, (num %10), i);
			}
			wait_write_bitstream(param);
			num = rand();
			for (i = 0; i< num % MAX_PRINT_NUM;i++) {
				printf("mem write seq=%d max=%d counter=%d\n", slice.seq, (num %10), i);
			}
			printf("slice end %d\n", slice.seq);
			if (slice.end == 1) {
				printf("end of frame\n");
				pthread_mutex_unlock(&end_frame_mutex);
			} else {
				start_write_next_bitstream(param);
			}

			counter++;



	}
	
	pthread_exit(NULL);
}

pthread_t thread[MAX_THREAD_NUMBER];

struct thread_param params[MAX_THREAD_NUMBER];

int thread_fd[MAX_THREAD_NUMBER];

void frame_end_mutex_init(void) {
	int ret;
	ret = pthread_mutex_init(&end_frame_mutex,NULL);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	pthread_mutex_lock(&end_frame_mutex);
}
void frame_end_wait(void) {	
	pthread_mutex_lock(&end_frame_mutex);
}


int encoder_thread_init(void)
{
	int i,ret;

	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		ret = pthread_mutex_init(&params[i].my_thread_mutex,NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}

		ret = pthread_mutex_lock(&params[i].my_thread_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		ret = pthread_mutex_init(&params[i].write_bitstream_my_mutex,NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}

		ret = pthread_mutex_lock(&params[i].write_bitstream_my_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
//	printf("%d\n", __LINE__);
	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		params[i].write_bitstream_next_mutex = &params[(i+1)%MAX_THREAD_NUMBER].write_bitstream_my_mutex;
	}
//	printf("%d\n", __LINE__);

	pthread_attr_t attr;
	ret = pthread_attr_init(&attr);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return -1;
	}
	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		params[i].thread_no = i;

		ret = pthread_create(&thread[i], &attr, &thread_start_routin, (void*)&params[i]);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		params[i].thread_no = i;
	}

	for(i=0;i<MAX_THREAD_NUMBER;i++) {
		char fifoname[1024];
		sprintf(fifoname, "/tmp/fifo%d", i);	
		mkfifo(fifoname, 0666);
		thread_fd[i] = open(fifoname, O_WRONLY);
	}
	frame_end_mutex_init();

	return 0;
}

void start_write_bitstream(void) {
		pthread_mutex_unlock(&params[0].write_bitstream_my_mutex);
}

int main()
{

	int i,j,ret;
	ret = encoder_thread_init();
	if (ret != 0) {
		printf("%d", __LINE__);
		return -1;
	}
	printf("encode start\n");
	for ( i=0;i<FRAME_NUM;i++) {
		printf("frame start\n");
		max_seq_number = MAX_SEQ_NUMBER;
		for(j=0;j<max_seq_number;j++ ) {
			struct Slice slice;
			slice.seq = j;
			if (j==(max_seq_number -1)) {
				slice.end = 1;
			} else {
				slice.end = 0;
			}
			write(thread_fd[j%MAX_THREAD_NUMBER], &slice, sizeof(slice));
		}

		start_write_bitstream();
		//wait thread
		printf("wait threads\n");
		frame_end_wait();
	}

	printf("main end\n");
	return 0;


}