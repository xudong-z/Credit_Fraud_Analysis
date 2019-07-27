library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)
library(shiny)
library(DT)

library(randomForest)
library(caret)
library(ROCR)
library(DMwR)
library(data.table)
library(zoo)


df_for_tab1 <- fread("creditcard.csv")[,-1]# for small 
df <- fread("creditcard.csv")[1:50000,-1] # for time saving
df_new <- fread("df_new.csv")[1:50000,-1] # for time saving
df_new$Class = as.factor(df_new$Class)
results <-  read.csv("all_results.csv")
strategy <- data.frame(
    Subset = c("Small Amout", "Large Amount"), Tolerance_Lelve = c("High", "Low"), 
    Detection_Strategy = c('Conservative', 'Radical'), Cutoff = c("High","Low"), Imortant_Metric = c('Precision', 'Recall'))


colnames(strategy)[1] = ""

df_for_tab1$Class_YN = ifelse(df_for_tab1$Class==0, "Non-Fraud", "Fraud")

df_for_tab1$Amount_cat <- ifelse(df_for_tab1$Amount == 0, "0",
                        ifelse(df_for_tab1$Amount <= 10, "(0,10]",
                               ifelse(df_for_tab1$Amount <=25, "(10,25]",
                                      ifelse(df_for_tab1$Amount <= 50, "(25-50]",
                                             ifelse(df_for_tab1$Amount <= 125, "(50,125]",
                                                    ifelse(df_for_tab1$Amount <= 250, "(125,250]", 
                                                           ifelse(df_for_tab1$Amount <= 500, "(250,500]", 
                                                                  ifelse(df_for_tab1$Amount <= 1000, "(500,1000]", 
                                                                         ifelse(df_for_tab1$Amount <= 2000, "(1000,2000]", '>2000')))))))))

df_for_tab1$Amount_cat = ordered(df_for_tab1$Amount_cat,
                        levels = c( "0", "(0,10]", "(10,25]",                                                  "(25-50]", "(50,125]","(125,250]","(250,500]","(500,1000]","(1000,2000]", '>2000'))
#table(df_for_tab1$Amount_cat,df_for_tab1$Class)

# calculate Area under PRC
auprc <- function(pr_curve) {
    x <- as.numeric(unlist(pr_curve@x.values))
    y <- as.numeric(unlist(pr_curve@y.values))
    y[is.nan(y)] <- 1
    id <- order(x)
    result <- sum(diff(x[id])*rollmean(y[id],2))
    return(result)
}

# UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI # UI 
ui <- fluidPage(tabsetPanel(
    
    tabPanel("1. Exploratory Data Analysis",
             titlePanel("Credit Fraud Data Analysis", # set this as the project title
                        windowTitle = "Xudong ZHANG/nJan 20 2018"),
             sidebarPanel(
                 helpText("Check deviation and distribution for each variable"),
                 selectInput(inputId = "tab1_var_select",
                              label = "Choose a predictor",
                              choices = colnames(df)[1:29],
                              selected = "V1"),
                 h3(""),
                 helpText("Check density distribution in details by amount shrinking"),
                 sliderInput(inputId = "tab1_amount_slide",label="Amount upper limit", 
                             min=5, max=2500, value=500, step=5),
                 h3(""),
                 helpText("Check distribution of different amount categories"),
                 radioButtons(inputId = "tab1_amount_cat",
                              label = "Choose a Subset of Amount Catogories",
                              choices = c("Fraud", "Non-Fraud"),
                              selected = "Fraud"),

                 width = 3),
             mainPanel(
                 plotOutput(outputId = "tab1_boxplot"),
                 splitLayout(plotOutput(outputId = "tab1_amount_slide_plot"),plotOutput(outputId = "tab1_amount_cat_plot")))),
    
    tabPanel("2. Evaluation and Model Selection",
             sidebarPanel(
                 helpText("Data Partition Proportion (train/test)"),
                 checkboxGroupInput(inputId = "tab2_parti",
                                    label = "Choose one or more sampling methods",
                                    choiceNames =list("50%Train / 50%Test",
                                                      "60%Train / 40%Test",
                                                      "70%Train / 30%Test",
                                                      "80%Train / 20%Test"),
                                    choiceValues = c("0.5","0.6", "0.7", "0.8"),
                                    selected = "0.6"),
                 width = 3),
             mainPanel(
                 tabsetPanel(id = "Visualized Results",
                             tabPanel("Visualized Results", 
                                      splitLayout(plotOutput(outputId = "tab2_AU_PRC"), plotOutput(outputId = "tab2_AU_ROC")),
                                      plotOutput(outputId = "tab2_avg_bar")),
                             tabPanel("All Scores List",
                                         dataTableOutput(outputId = "tab2_table_results"))))),
    
    tabPanel("3. Detection Strategy Analysis",
             mainPanel(
                 h2("Fraud Detection Strategy"),
                 tableOutput(outputId = "tab3_strategy"),
                 plotOutput(outputId = "tab3_rf_pr"))),
    
    tabPanel("4. Modeling on Different Amount Level",
             sidebarPanel(
                 helpText("RF Tuning for Small&Large Amount Transanctions"),
                 sliderInput(inputId = "tab4_parti", label = "--Data Partitioning (training set)", min=0, max=1, value=0.6, step=0.1),
                 sliderInput(inputId = "tab4_SMOTE", label = "--SMOTE Sampling",min=100, max=1000, value=400, step=50),
                 sliderInput(inputId = "tab4_amount",label = "--Amount Threshold", min=0, max=1000, value=400, step=25),
                 splitLayout(sliderInput(inputId = "tab4_precision_small", label = "Precision for Small", min=0, max=1, value=0.60, step=0.01),
                 sliderInput(inputId = "tab4_recall_large", label = "Recall for Large", min=0, max=1, value=0.90, step=0.01)),
                 width = 3),
             mainPanel(
                 #textOutput("tab4_test"),
                 splitLayout(plotOutput(outputId = "tab4_pr_small"), plotOutput(outputId = "tab4_pr_large")),
                 tableOutput(outputId = "tab4_results"))),
    tabPanel("5. Actionable Suggestions",
             sidebarPanel(width = 1),
             mainPanel(
             h2("1. Best model recommendation:"),
             h3("   -- Random Forest"),
             h3("   -- Partition = 0.6"),
             h3("   --SMOTE.perc.over = 200"),
             h2("2. Best Cutoff recommendation:"),
             h3("   --Amount threshold = 250"),
             h3("   --Small Amount cutoff = 0.72"),
             h3("   --Large Amount cutoff = 0.52"),
             h2("3. Try more exploratory analysis if given time-series data"),
             h2("4. Try Neural Network if given a stronger CPU!")
             ))
    ))

# Server # Server # Server # Server # Server # Server # Server # Server # Server # Server # Server 
server <- function(input, output) {
    df_for_tab1_r <- reactive(df_for_tab1)
    df_r <- reactive({df})
    #df_new_r <- reactive({df_new})
    results_r <- reactive({results})
    strategy_r <- reactive({strategy})
    
    # tab1 EDA
    output$tab1_boxplot <- renderPlot({    # 先查看各X在两个Class下的deviation
        ggplot(data = df_for_tab1_r(), aes_string(x = 'Class', y = input$tab1_var_select)) +  # y.value  #aes_string
            geom_boxplot(aes(group = Class, fill = Class, alpha = .3))+
            labs(title = paste("Boxplot of", input$tab1_var_select,"By Fraud Class"))})
    output$tab1_amount_cat_plot <- renderPlot({  # 重点关注amount分布
        df_for_tab1_r() %>%
            filter(Amount_cat != "NA") %>%
            filter(Class_YN %in% input$tab1_amount_cat) %>%  # select CLASS
            ggplot(aes(x = Amount_cat)) + 
            geom_bar(stat = "count", position = 'dodge' ,aes(group = Class, fill = Class))+
            geom_label(aes(label=..count..),stat="count",position=position_stack())+
            labs(title = paste("Amount Category Distribution" ), x = " ")+
            theme(axis.text.x=element_text(angle=15, hjust=1), legend.position = "none")})
    output$tab1_amount_slide_plot <- renderPlot({
        ggplot(df_for_tab1_r()[Amount < as.numeric(input$tab1_amount_slide),], aes(x = Amount, group = Class, color = Class, fill = Class)) + # select Predictors
            stat_density( position="identity", alpha = .3) +
            labs(title = paste("Density Distribution By Fraud Class (Amount <",as.numeric(input$tab1_amount_slide),")" ))})
        
    output$tab2_AU_PRC <- renderPlot({
        results %>%
            filter(! Model %in% c("LASSO", "RIDGE")) %>%
            filter(Data_Patition %in% input$tab2_parti) %>%
            group_by(Model, SMOTE_OverPercent) %>%
            summarise(AVG_AU_PRC = round(mean(AU_PRC),4)) %>%
            ggplot(aes(x = SMOTE_OverPercent, y = AVG_AU_PRC, 
                       group = Model, color = Model, label = AVG_AU_PRC))+
            geom_line()+ geom_text()+ 
            scale_x_continuous(limits = c(175, 825)) +
            scale_y_continuous(limits = c(0, 0.8)) +
            labs(x = "SMOTE Oversampling Percent Level", y = " ", 
                 title = "Score of AU PR Cureve Between Different SMOTE Levels")+
            theme(legend.position= "bottom")})
    
    output$tab2_AU_ROC <- renderPlot({
        results %>%
            filter(! Model %in% c("LASSO", "RIDGE")) %>%
            filter(Data_Patition %in% input$tab2_parti) %>%
            group_by(Model, SMOTE_OverPercent) %>%
            summarise(AVG_AU_ROC = round(mean(AU_ROC),4)) %>%
            ggplot(aes(x = SMOTE_OverPercent, y = AVG_AU_ROC, 
                       group = Model, color = Model, label = AVG_AU_ROC))+
            geom_line()+
            geom_text()+
            scale_x_continuous(limits = c(175, 825)) +
            scale_y_continuous(limits = c(0.92, 1))+
            labs(x = "SMOTE Oversampling Percent Level", y = " ", 
                 title = "Score of AU ROC Between Different SMOTE Levels")+
            theme(legend.position = "bottom")})
    
    output$tab2_avg_bar <- renderPlot({
        results %>%
            filter(! Model %in% c("LASSO", "RIDGE")) %>%
            filter(Data_Patition %in% input$tab2_parti) %>%
            group_by(Model, SMOTE_OverPercent) %>%
            mutate(AVG_PRC = round(mean(AU_PRC),4)) %>%
            group_by(Model, SMOTE_OverPercent) %>%
            mutate(AVG_ROC = round(mean(AU_ROC),4)) %>%
            gather("Type", "Avg", 7:8) %>% 
            distinct(Model, Type, Avg) %>%
            group_by(Model, Type) %>%
            summarise(Avg_AreaUnderCurve = round(mean(Avg),4)) %>%
            ggplot(aes(x = Model, y = Avg_AreaUnderCurve, fill = Type, label = Avg_AreaUnderCurve)) +
            geom_bar(stat = "identity", position = "dodge")+
            #geom_text()+
            geom_label(stat="identity")+
            labs(x = "", y = "Average Score of AUC", 
                 title = "Average Area Under Curve of Four SMOTE's Modeling") +
            #theme(legend.position = "bottom") + 
            scale_fill_discrete(labels=c("PR Curve", "ROC")) })
    output$tab2_table_results <- DT::renderDataTable({
        results_r()
    })
## -----------------------Above Not training---------------------------------##    
    output$tab3_strategy <- renderTable(
        strategy_r(), spacing = c("l"), width = 50)
    
    output$tab3_rf_pr <- renderPlot({plot(rf_pr_small())})
    
## -----------------------Tab 4 training---------------------------------##   
    small_index<- reactive({df$Amount < input$tab4_amount}) # Note: df_new has been centerized
    df_small <- reactive({df_new[small_index(),]})
    df_large <- reactive({df_new[-small_index(),]})
    
    
    ### SMALL AMOUNT
    training_small <- reactive({df_small()[sample(nrow(df_small()), input$tab4_parti*nrow(df_small())),]})  # correct 已出
    test_small <- reactive({df_small()[-sample(nrow(df_small()), input$tab4_parti*nrow(df_small())),]})
    training_SMOTE_small <- reactive({   # 问题出在这 !!!!????
        SMOTE(as.factor(Class)~., training_small(), perc.over=input$tab4_SMOTE, perc.under=100)}) # 可以在reactive之前就factor
    rf_small <- reactive({
        randomForest(as.factor(Class) ~ ., data = training_SMOTE_small(), 
                     ntree = 500, nodesize = 20, mtry = 13, importance = TRUE,
                     na.action = na.pass)})
    rf_pr_small <- reactive({
        predict(rf_small(), test_small(), type="prob")[,2] %>%
        prediction(test_small()$Class) %>%
        performance("prec","rec")})
    rf_arprc_small = reactive(auprc(rf_pr_small()))
    rf_pr_s_df = reactive({
        data.frame(cutoff = rf_pr_small()@alpha.values[[1]], recall = rf_pr_small()@x.values[[1]], precision = rf_pr_small()@y.values[[1]])})
    # tuning cutoff SMALL
    s_cutoff_precision = reactive({input$tab4_precision_small}) #input$tab4_precision_small
    s_cutoff= reactive({
        as.numeric(rf_pr_s_df() %>% filter(precision >=s_cutoff_precision()) %>% 
                       arrange(precision) %>% slice(1) %>% select(cutoff))}) # as.numerica 防止过滤出df
    s_cutoff_recall = reactive({ 
        as.numeric(rf_pr_s_df() %>% filter(precision>=s_cutoff_precision()) %>% 
                       arrange(precision) %>% slice(1) %>% select(recall))})
    output$tab4_pr_small <- renderPlot({
        ggplot(rf_pr_s_df(), aes(x=recall, y = precision))+
        geom_line() +
        geom_hline(yintercept = s_cutoff_precision(), color = 'grey') +
        geom_point(aes(x = s_cutoff_recall(), y = s_cutoff_precision()),show.legend = T)+
        xlim(0.5,1)+ylim(0,1)+labs(title = "Precision-Recall Curve for Small Amount",
                                   subtitle = "(Precision-Oriented, Conservative)")
    })
    
    #df_new$Class = as.factor(df_new$Class)########################----------------########################
    #df_large <- reactive({subset(df_new, Amount >= input$tab4_amount)})
    training_large <- reactive({df_large()[sample(nrow(df_large()), input$tab4_parti*nrow(df_large())),]})  # correct 已出
    test_large <- reactive({df_large()[-sample(nrow(df_large()), input$tab4_parti*nrow(df_large())),]}) 
    training_SMOTE_large <- reactive({   # 问题出在这 !!!!????
        SMOTE(as.factor(Class)~., training_large(), perc.over=input$tab4_SMOTE, perc.under=100)}) # 可以在reactive之前就factor
    rf_large <- reactive({
        randomForest(as.factor(Class) ~ ., data = training_SMOTE_large(), 
                     ntree = 500, nodesize = 20, mtry = 13, importance = TRUE,
                     na.action = na.pass)})
    rf_pr_large <- reactive({
        predict(rf_large(), test_large(), type="prob")[,2] %>%
            prediction(test_large()$Class) %>%
            performance("prec","rec")})
    rf_arprc_large = reactive(auprc(rf_pr_large()))
    rf_pr_l_df = reactive({
        data.frame(cutoff = rf_pr_large()@alpha.values[[1]], recall = rf_pr_large()@x.values[[1]], precision = rf_pr_large()@y.values[[1]])})
    # tuning cutoff LARGE
    l_cutoff_recall = reactive({input$tab4_recall_large}) 
    l_cutoff_precision = reactive({
        as.numeric(rf_pr_l_df() %>% filter(recall >= l_cutoff_recall()) %>%
                   arrange(recall) %>% slice(1) %>% select(precision))}) # as.numerica 防止过滤出df
    l_cutoff = reactive({ 
        as.numeric(rf_pr_l_df() %>% filter(recall >= l_cutoff_recall()) %>% 
                       arrange(recall) %>% slice(1) %>% select(cutoff))}) #匹配不上 
    output$tab4_pr_large <- renderPlot({
        ggplot(rf_pr_l_df(), aes(x=recall, y = precision))+
            geom_line() +
            geom_vline(xintercept = l_cutoff_recall(), color = 'grey') +
            geom_point(aes(x = l_cutoff_recall(), y = l_cutoff_precision()),show.legend = T)+
            xlim(0.6,1)+ylim(0,1)+labs(title = "Precision-Recall Curve for Large Amount", 
                                       subtitle = "(Recall-Oriented, Radical)")})
    
    tab4_results <- reactive({
        data.frame(" " = c("Cutoff", "Precision", "Recall", "Area Under PR"), 
                   "Small Amount(L)" = c(s_cutoff(),s_cutoff_precision(),s_cutoff_recall(),rf_arprc_small()),
                   "Large Amount(R)" = c(l_cutoff(),l_cutoff_precision(),l_cutoff_recall(),rf_arprc_large()))})
    output$tab4_results <- renderTable(
        tab4_results(), spacing = c("l"), width = 24 )
    #output$tab4_test <- renderText({c(dim(df_large()),dim(df_small()))})
}

# run
shinyApp(ui = ui, server =server)