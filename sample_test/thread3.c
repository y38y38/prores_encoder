#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#define MAX_SEQ_NUMBER 	(13)
#define FRAME_NUM (3)

int max_seq_number = 0;
#define MAX_THREAD_NUM (4)

#define MAX_PRINT_NUM	(10)

struct Slice {
	int seq;
//	struct Slice *next;
	int end;
};

struct Slice slice_param[MAX_SEQ_NUMBER];

struct thread_param {
//	int seq;
	int thread_no;
//	pthread_cond_t  my_thread_cond;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
//	struct Slice *top;
};

pthread_mutex_t slice_num_thread_mutex[MAX_THREAD_NUM];
pthread_cond_t slice_num_thread_cond[MAX_THREAD_NUM];
int slice_num_thread[MAX_THREAD_NUM];



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

#if 0
	char fifoname[1024];
	sprintf(fifoname, "/tmp/fifo%d", param->thread_no);	
	mkfifo(fifoname, 0666);

	int fd = open(fifoname, O_RDONLY);
#endif
	
	int counter = 0;
	for(;;) {
		struct Slice *slice;
#if 0
		read(fd, &slice, sizeof(slice));
#endif
		pthread_mutex_lock(&slice_num_thread_mutex[param->thread_no]);
		while(slice_num_thread[param->thread_no] == 0) {
			counter = 0;
			pthread_cond_wait(&slice_num_thread_cond[param->thread_no], &slice_num_thread_mutex[param->thread_no]);
		}
		printf("s slice_num_thread %d %d\n", param->thread_no, slice_num_thread[param->thread_no]);

		pthread_mutex_unlock(&slice_num_thread_mutex[param->thread_no]);

			int i;
			int index = (counter * MAX_THREAD_NUM) + param->thread_no;
			int num = rand();
			for (i = 0; i< num % MAX_PRINT_NUM;i++) {
				printf("dct seq=%d max=%d counter=%d\n", slice_param[index].seq, (num %10), i);
			}
			wait_write_bitstream(param);
			num = rand();
			for (i = 0; i< num % MAX_PRINT_NUM;i++) {
				printf("mem write seq=%d max=%d counter=%d\n", slice_param[index].seq, (num %10), i);
			}
			printf("slice end %d\n", slice_param[index].seq);
			if (slice_param[index].end == 1) {
				printf("end of frame\n");
				pthread_mutex_unlock(&end_frame_mutex);
			} else {
				start_write_next_bitstream(param);
			}

			counter++;
		pthread_mutex_lock(&slice_num_thread_mutex[param->thread_no]);
		slice_num_thread[param->thread_no]--;
		printf("e slice_num_thread %d %d\n", param->thread_no, slice_num_thread[param->thread_no]);
		pthread_mutex_unlock(&slice_num_thread_mutex[param->thread_no]);


	}
	
	pthread_exit(NULL);
}

pthread_t thread[MAX_THREAD_NUM];

struct thread_param params[MAX_THREAD_NUM];

int thread_fd[MAX_THREAD_NUM];

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

	for(i=0;i<MAX_THREAD_NUM;i++) {
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
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].write_bitstream_next_mutex = &params[(i+1)%MAX_THREAD_NUM].write_bitstream_my_mutex;
	}

//	printf("%d\n", __LINE__);
	for(i=0;i<MAX_THREAD_NUM;i++) {
		ret = pthread_mutex_init(&slice_num_thread_mutex[i],NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		ret = pthread_cond_init(&slice_num_thread_cond[i],NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		slice_num_thread[i] = 0;
	}

	pthread_attr_t attr;
	ret = pthread_attr_init(&attr);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return -1;
	}
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].thread_no = i;

		ret = pthread_create(&thread[i], &attr, &thread_start_routin, (void*)&params[i]);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].thread_no = i;
	}
#if 0
	for(i=0;i<MAX_THREAD_NUM;i++) {
		char fifoname[1024];
		sprintf(fifoname, "/tmp/fifo%d", i);	
		mkfifo(fifoname, 0666);
		thread_fd[i] = open(fifoname, O_WRONLY);
	}
#endif

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
//			struct Slice slice;
			slice_param[j].seq = j;
			if (j==(max_seq_number -1)) {
				slice_param[j].end = 1;
			} else {
				slice_param[j].end = 0;
			}
			//write(thread_fd[j%MAX_THREAD_NUM], &slice, sizeof(slice));
		}
		for(j=0;j<MAX_THREAD_NUM;j++) {
			pthread_mutex_lock(&slice_num_thread_mutex[j]);
			if ((max_seq_number % MAX_THREAD_NUM) > j) {
				slice_num_thread[j] = (max_seq_number / MAX_THREAD_NUM) + 1;
			} else {
				slice_num_thread[j] = (max_seq_number / MAX_THREAD_NUM);
			}
			printf("m slice_num_thread %d %d\n", j, slice_num_thread[j]);
			pthread_cond_signal(&slice_num_thread_cond[j]);
			pthread_mutex_unlock(&slice_num_thread_mutex[j]);

		}

		start_write_bitstream();
		//wait thread
		printf("wait threads\n");
		frame_end_wait();
		for(j=0;j<MAX_THREAD_NUM;j++) {
			printf("%d\n", slice_num_thread[j]);
		}
	}

	printf("main end\n");
	return 0;


}