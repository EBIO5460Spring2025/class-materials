### Homework to get started

See also reading for next week. This homework is not assessed. Aim to finish everything before class on Tuesday. If you're having Git troubles, ask on Piazza, and help each other out. 



#### 1. Email me your Github username

If you don't already have a GitHub account, sign up for one. Email me your GitHub username so I can add you to the class organization. This is where you will submit assignments.



#### 2. Git & GitHub refresher or intro

If you are new to Git and GitHub, or you need a git refresher, you can use the tutorials from fall 2024 [here](https://github.com/EBIO5460Fall2024/class-materials/tree/main/skills_tutorials). Relevant sections are:

* [git00_resources.md](https://github.com/EBIO5460Fall2024/class-materials/blob/main/skills_tutorials/git00_resources.md)
* [git01_setup.md](https://github.com/EBIO5460Fall2024/class-materials/blob/main/skills_tutorials/git01_setup.md) (tailored to R; for Python users see git00_resources)
* [git03_basics.md](https://github.com/EBIO5460Fall2024/class-materials/blob/main/skills_tutorials/git03_basics.md)
* [git04_amend.md](https://github.com/EBIO5460Fall2024/class-materials/blob/main/skills_tutorials/git04_amend.md)
* [git06_gitgui.md](https://github.com/EBIO5460Fall2024/class-materials/blob/main/skills_tutorials/git06_gitgui.md)




#### 3. Take control of your GitHub repo for this class

After you send me your GitHub username, I'll set up a GitHub repo for you that is within the private GitHub space for this class. I'll email you when it's ready to go. This repo is not public (i.e. not open to the world). You and I both have write access to this repo. Clone it to your computer using whatever method you prefer. For example, if you're using Git via RStudio :

1. Go to the class GitHub organization: https://github.com/EBIO5460Spring2025.
2. Find your repo. It should be visible on the `Repositories` tab. Your repo is called `ml4e_firstnamelastinitial`.
3. From the green `Code` button in your repo on GitHub, copy the repo's URL to the clipboard.
4. Clone the repo to an RStudio project on your computer.
   1. File > New Project > Version Control > Git.
   2. In `Repository URL`, paste the URL; leave `Project directory name` blank; browse to where you want to put it; click `Create Project`.


In the repository you just cloned, you'll find three files (if you used the RStudio method above):

1. A file with extension `.Rproj`. This file was created by RStudio. To open your project in RStudio, double click the file's icon. When RStudio opens, you'll be in the working directory for the project.
2. `README.md`. This file was created by GitHub. This is a plain text file with extension `.md`, indicating that it is a file in Markdown format. You can edit this file using a text editor.
3. `.gitignore`. This file was created by RStudio but it is a file used by Git. This file tells Git which files or types of files to ignore in your repository (i.e. files that won't be under version control). By default, RStudio tells Git to ignore several files including `.Rhistory` and `.Rdata` because it usually doesn't make sense to track these temporary files. You can use a text editor to add other files to `.gitignore`.

If you used a different method to clone your repository, you'll only find the `README.md` file.

