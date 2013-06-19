---
published: false
layout: post
comments: true
title: Hadoop MapReduce Cookbook. Simple analytics using MapReduce
date: "2013-05-28T15:00:0"
categories: Java Hadoop Map-Reduce worked-example
---

In this post I work through the nasty details of getting the [Hadoop MapReduce Cookbook](http://srinathsview.blogspot.com/2013/01/our-book-hadoop-mapreduce-cookbook-is.html)'s Chapter 6 example, "Simple analytics using MapReduce", up and running.  I will generally discuss only those steps that differ from the authors' published instructions.

I'm using a managed, virtual Hadoop cluster on CentOS Linux, and I do not have root privileges.  As a result, I can only write to my home user space on the HDFS, and not / or HADOOP_HOME.  I have to do everything locally.

The recipe...

(1) NTR (nothing to report)

(2) I don't have access to the root of the HDFS, so I cannot create a directory /data, so I used my home directory.
{% highlight bash %}
> hadoop fs -mkdir data
> hadoop fs -mkdir data/input1
> hadoop fs -put <DATA_DIR>/NASA_access_log_Jul95 data/input1
{% endhighlight %}

(3) NTR

(4) NTR

(5) I don't have Apache Ant, so I did a local installation from source.  Ant requires [Junit](http://junit.org/), so I put junit-4.11.jar and hamcrest-core-1.3.jar into apache-ant-1.9.1/lib/optional.  I then built Ant.
{% highlight bash %}
> sh build.sh -Ddist.dir=$HOME/local dist
{% endhighlight %}
And I added $HOME/local/bin to my path.  The ant build of Chapter 6 code then went fine.

(6) I don't have write privileges to HADOOP_HOME, so I skipped this step.

(7) I ran this as
{% highlight bash %}
> hadoop jar build/lib/hadoop-cookbook-chapter6.jar chapter6.WebLogMessageSizeAggregator data/input1 data/output1
{% endhighlight %}
Note that I have a space after "Aggregator".

(8) NTR

Fin.
