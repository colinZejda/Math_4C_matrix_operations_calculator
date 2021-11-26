const tabs = document.querySelectorAll('[data-tab-target]')
const tabContents = document.querySelectorAll('[data-tab-content]')      //new variable tabContents, this will contain all the contents for our tabs
tabs.forEach(tab => {                                                    //loops thru each tab
    tab.addEventListener('click', () => {
        const target = document.querySelector(tab.dataset.tabTarget)     //gets us our home element with each tab we click(it's a function)
        
        tabContents.forEach(tabContent => {        //loop over each tab's tabContent, this makes sure the only the tab we click on is active(aka showing on the website), otherwise, all the tab info will pop up when clicked, but not go away
            tabContent.classList.remove('active')
        })
        target.classList.add('active')                                   //to make the target visible, we'll use CSS to style this class 
        
        tabs.forEach(tab => {               //here, we loop over each tab
            tab.classList.remove('active')
        })
        tab.classList.add('active')
        target.classList.add('active')
    })
});