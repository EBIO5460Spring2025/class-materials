# How to run R on a server
Before we get on to detailed instructions, here are the most needed commands (from the Linux prompt):
```bash
nohup R --vanilla < infile.R > out.txt &
top #table of processes; hit q to quit
ls #list files in directory
cat out.txt #Print the file out.txt to the screen.
```



## Login and setup

For most servers, including supercomputer resources, you'll login to the server via ssh. Do this from a terminal on your desktop or laptop. Mac OS has a built in Terminal application generally found in the Utilities folder in Applications. On Windows you can use the built in Powershell terminal. Linux distributions will have at least one terminal option. You'll need instructions from the server administrator but typically it will be something like

```bash
ssh jbsmith@login.rc.colorado.edu
```

For university servers you may need to VPN to the campus network first.

It's also useful to have a GUI app to transfer files back and forth from your laptop to the server. You can then shuttle files back and forth with ease, in near real time, to any directory on the server. This allows you to prototype on your local machine, push the file to Linux on the server, run R, and pull results files back to the local machine for post-processing/graphing. You can also transfer files with command line tools but it's much faster and more convenient to work with a GUI.

* [WinSCP](http://winscp.net) (opensource). Win (recommended).
* [Cyberduck](https://cyberduck.io/) (opensource). Win, Mac (recommended).
* [Filezilla](https://filezilla-project.org/) (opensource). Win, Mac, Linux.

In WinSCP and Filezilla the file browser has two window panes, one showing files on the client, the other showing files on the server. You can manually transfer files back and forth or set it to synchronize client and host. Cyberduck presents a browser for the server's file system and you can drag and drop files to/from there, or set it to synchronize a folder.



## File management and the R working directory

When you login to the server you'll be in your home directory, e.g. `/home/brett`, which is indicated by the `~` symbol. If you transferred files from your Mac or Windows computer this is where they probably are. To list the files in a directory type:
```bash
ls
```

You can run R from here, or you can make new directories (folders) within your home directory. To make a new directory
```bash
mkdir mydirectory
```
You can move files using
```bash
mv myfile mydirectory
```

Navigate to the working directory before starting R.
```bash
cd mydirectory       #change directory
```
n.b. it's only necessary to `setwd("/home/brett/mydirectory")` in the R program if you run it from a different directory. Avoid using `setwd()` - it leads to code that is not reproducible or transferable between computers.

Some other useful linux commands:
```bash
rm myfile            #delete (remove) files
cp myfile mynewfile  #copy files
rmdir mydirectory    #remove directory
rm -r mydirectory    #remove directory with all its subfolders and files
man <command>        #help (manual) on linux commands
man rm               #e.g. help for remove
```



## Start an interactive R session

This is a command line session only. See later for options for graphics sessions.

```bash
R
q() #To quit the session
```

You can copy code chunks from your local machine and paste them into the server session. This way you can do compute intensive parts on the server and graphics locally. You can easily transfer results from the server to your local computer like this:
```R
save(myobject,file="out.Rdata") #run this line on server, or
save.image(file="out.Rdata") #for the whole environment.
load("out.Rdata") #After transferring file, load the object(s) on your local computer with this.
```

While you can't view graphs from this command line session, you can make graphs in various formats that are written to file, such as .png, or .pdf.

This isn't the most efficient way to work but sometimes it's all you have access to when you're working on someone else's computer (e.g. a supercomputing resource).



## Install R packages

Install packages from within an interactive R session, e.g.

```R
install.packages("vegan")  #install the vegan package
```



## Run an R script from the command line

Here, we are running a pre-written script all at once. For a longer job, this is preferred to an interactive session.

### Variation 1
```bash
R --vanilla < infile.R &
```
This will set the program going and the results will print to the terminal. The symbol `<` redirects the contents of the file `infile.R` to the R process. The `&` puts the process into the background, `--vanilla` directs R to run quietly and exit without saving the workspace.

### Variation 2
```bash
nohup R --vanilla < infile.R &
```
`nohup` stands for "no hang up" and redirects output to the file `nohup.out` by default.

You can now logout as usual and the process will keep running. If you omit `nohup` the process will quit if you logout.

### Variation 3
What I use most of the time
```bash
nohup R --vanilla < infile.R > out.txt &
```
The symbol `>` redirects the output (from stdout, i.e. what would normally print to the R console) to a file called `out.txt`. You can inspect this file to check that everything ran correctly. Indeed, you can inspect this file any time while the program is running, so you can monitor progress, e.g.
```bash
cat out.txt
tail out.txt
```

At this point R will start running and you'll get a process ID returned at the terminal. It's handy to write this ID down in case you need to kill the process.

The prompt will now say `nohup: redirecting stderr to stdout`
Press enter to return to a normal prompt.

It's also worth saving results, or the workspace, within R before your program quits, e.g. add one of these lines to your code:
```R
save(resultvector,file="results.Rdata") #saves a particular object
save.image(file="workspace.Rdata")      #saves the whole workspace
```



## Checking and managing running jobs

### Check that R is running

Show all the processes running on the server and the resources they're using:
```bash
top #table of processes
```
Process ID is also given here. Type `q` to quit.

Show just processes from one user and their process IDs:
```bash
ps -u brett
```
This is useful for checking ones that are running but not using CPU time. If you have a parallel R job running, there will be one process for the controller and one for each of the workers. The first one is the controller and it will not be using any significant CPU time so will infrequently show using `top`.


### Stop a computation
For when you're in an R session in the terminal window.

Sometimes you need to stop a running computation (e.g. it's taking too long, or you got in an infinite loop). While within R, use

```R
Ctrl-c
```
This should return you to the R prompt.

Sometimes this doesn't work and you have to take more drastic action. Hit `Break` or `Ctrl-z`. This will suspend R altogether. Now you are not in R any more and you need to kill the suspended R. When a program is stopped and the cursor returned to the linux prompt, the job number is given in square braces `[]`. Use this number to kill the job (see kill a job or process below). If you don't see this number, type
```bash
jobs
```

### Stop a parallel computation
If you want to stop a parallel R job that you started from a script, then you need to kill all the processes that were started by it (both controller and workers). There are three ways.

1) If you're still in the terminal session where you started the job (i.e. you have not logged out and logged back in again), find out the job number and use that to kill the job.
```bash
jobs
```
The job number is the number in the square braces `[]`. This will kill the controller and all the workers and is the safest and most convenient way.

2) If you have in the meantime logged out and back in, then you can't get a job number and need to kill the processes. Find the process IDs of your running R jobs using the following command (substitute your user name):
```bash
pgrep -au brett R
```
The controller process will generally be the first one (with `--vanilla` in its command call) and the workers are identified by `--slave` in their command calls.

If you want to kill all of these, then do this
```bash
pkill -u brett R
```
(by default `pkill` will send a `SIGTERM` signal to terminate the processes).

If you only want to kill some of them (e.g. if you have multiple jobs running but you only want to kill some of them), then you'll need to kill the processes one by one. Note their process IDs from
```bash
pgrep -au brett R
```
Then kill them individually (see kill a job or process below).

3) Kill the controller process. The worker processes will eventually terminate by themselves but they will continue running at 100% CPU time until they finish their calculation. In testing, I found that they terminate a few seconds after completing their calculation. Identify the controller process as in (2). This is also the PID that is printed when you first start the R script, so it's handy to write it down.

Finally, check that the processes have really gone. The following command will provide a lot of information on all of the processes currently on the server, including other users.
```bash
ps axu
```
You can narrow in to R processes using
```bash
ps -FC R
```
or (although this will return any line of output with the string "R")
```bash
ps axu | grep R
```
If the processes did not go away, kill with `SIGKILL` (see next). 


### Kill a job or process
```bash
kill %<num>
```
where num is job number or 
```bash
kill <num>
```
where num is the process id (don't use % for a process).

By default, `kill` sends a `SIGTERM` signal to the process, which causes it to quit gracefully. If this doesn't work, use:
```bash
kill -SIGKILL <num>
```
which will force it.

You can kill multiple processes at once, e.g.
```bash
kill 4932 4944 4952
```


### Pause a job or process
This can be handy if you need to pause a long job while you do another job. First use the techniques above to find the job or process IDs. The general idea is to use `kill` to send the `SIGSTOP` signal, then you can later resume with `SIGCONT`. e.g.
```bash
kill -SIGSTOP 7975 7984 7993 8002
kill -SIGCONT 7975 7984 7993 8002
kill -l #for signal options.
```
Given that the workers will die without a controller, pause the workers first. Then pause the controller. Reverse the process to restart (start controller first).

Here's an example of a complete stop and start process for a 12 core cluster.
```bash
pgrep -u brett R #copy output to your text editor to make the lists of the correct PIDs
kill -SIGSTOP 7975 7984 7993 8002 8011 8020 8029 8038 8047 8056 8065 8074
kill -SIGSTOP 7972

kill -SIGCONT 7972
kill -SIGCONT 7975 7984 7993 8002 8011 8020 8029 8038 8047 8056 8065 8074
```

